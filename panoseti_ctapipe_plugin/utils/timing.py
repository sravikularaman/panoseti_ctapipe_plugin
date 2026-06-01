"""
Timestamp conversion and timing synchronization functions for PANOSETI data processing.

This module contains functions for:
- White Rabbit to Unix timestamp conversion
- Timestamp extraction from PFF metadata
- Multi-telescope timing offset measurement and correction
- Telescope clock synchronization

Last modified: 22 April 2026
"""

import logging

import numpy as np
import astropy.units as u

__all__ = [
    "wr_to_unix",
    "extract_timestamps_from_metadata",
    "measure_telescope_timing_offset",
    "correct_telescope_timing",
]

logger = logging.getLogger(__name__)


def wr_to_unix(pkt_nsec, tv_sec, tv_usec, ignore_clock_desync=False):
    """
    Convert White Rabbit timestamps to unix time (datetime64[ns]).

    Handles clock desynchronization between White Rabbit and Unix clocks
    using heuristics to determine the correct second boundary.

    Parameters
    ----------
    pkt_nsec : int or array-like
        Packet nanoseconds (White Rabbit sub-second precision)
    tv_sec : int or array-like
        Unix seconds
    tv_usec : int or array-like
        Unix microseconds (fallback sub-second precision)
    ignore_clock_desync : bool
        If True, ignore clock desynchronization errors

    Returns
    -------
    np.ndarray
        Array of datetime64[ns] timestamps
    """
    # Cast to arrays
    pkt_nsec = np.asarray(pkt_nsec, dtype=np.int64)
    tv_sec = np.asarray(tv_sec, dtype=np.int64)
    tv_usec = np.asarray(tv_usec, dtype=np.int64)

    # Unit conversions using astropy
    ns_per_second = int((1 * u.s).to(u.ns).value)
    usec_to_ns = int((1 * u.us).to(u.ns).value)
    threshold_ns = int((25 * u.ms).to(u.ns).value)

    # Convert tv_usec to ns
    usec_ns = tv_usec * usec_to_ns

    # Difference between Unix microseconds and White Rabbit nanoseconds
    diff = usec_ns - pkt_nsec

    # Determine which case applies based on clock consistency
    mask0 = np.abs(diff) < threshold_ns  # Normal case: same second
    mask1023 = diff > threshold_ns  # tv_usec much larger: Unix second +1
    mask1 = diff < -threshold_ns  # pkt_nsec much larger: Unix second -1

    # Preallocate output as ns integer
    out_ns = np.empty_like(tv_sec, dtype=np.int64)

    # Normal case: same second boundary
    # Use pkt_nsec (White Rabbit) as sub-second precision source
    out_ns[mask0] = tv_sec[mask0] * ns_per_second + pkt_nsec[mask0]

    # Case 1: pkt_nsec >> tv_usec → Unix second is off by -1
    # Correct by using previous second with pkt_nsec offset
    out_ns[mask1] = (tv_sec[mask1] - 1) * ns_per_second + pkt_nsec[mask1]

    # Case 1023: tv_usec >> pkt_nsec → Unix second is off by +1
    # Correct by using next second with pkt_nsec offset
    out_ns[mask1023] = (tv_sec[mask1023] + 1) * ns_per_second + pkt_nsec[mask1023]

    # Handle bad cases (large mismatch)
    mask_bad = ~(mask0 | mask1 | mask1023)

    if np.any(mask_bad):
        if ignore_clock_desync:
            out_ns[mask_bad] = tv_sec[mask_bad] * ns_per_second + pkt_nsec[mask_bad]
        else:
            i = np.flatnonzero(mask_bad)[0]
            raise Exception(
                f"Clock mismatch: tv_sec={tv_sec[i]} tv_usec={tv_usec[i]} "
                f"pkt_nsec={pkt_nsec[i]} diff={diff[i]}"
            )

    # Convert to datetime64[ns]
    return out_ns.astype("datetime64[ns]")


def extract_timestamps_from_metadata(metadata, ignore_clock_desync=False):
    """
    Extract event timestamps from PFF metadata using legacy cut_pkt_loss_old semantics.

    Builds per-QUABO timestamps as ``tv_sec + tv_usec * 1e-6`` and returns
    the minimum timestamp across all QUABOs.

    Parameters
    ----------
    metadata : dict
        Metadata dictionary from PFF file containing quabo_0, quabo_1, quabo_2, quabo_3 entries.
        Each entry should have: pkt_nsec, tv_sec, tv_usec arrays
    ignore_clock_desync : bool, optional
        Present for API compatibility. Unused by the legacy extraction path.

    Returns
    -------
    timestamps : np.ndarray
        Array of float64 unix-second timestamps (minimum across QUABOs per event)
    valid_mask : np.ndarray
        Boolean array (True = keep, False = invalid timestamp < 2000-01-01)
    """
    timestamps_quabo = []
    for i in range(4):
        quabo = metadata[f"quabo_{i}"]
        ts = (
            np.asarray(quabo["tv_sec"], dtype=np.float64)
            + np.asarray(quabo["tv_usec"], dtype=np.float64) * 1e-6
        )
        timestamps_quabo.append(ts)

    timestamps = np.min(np.array(timestamps_quabo), axis=0)

    valid_threshold = np.datetime64("2000-01-01", "ns").astype("int64") * 1e-9
    valid_mask = timestamps >= valid_threshold
    if not np.all(valid_mask):
        n_invalid = np.sum(~valid_mask)
        logger.warning(f"Found {n_invalid} events with invalid timestamps (before 2000-01-01)")

    return timestamps, valid_mask


def measure_telescope_timing_offset(timestamps1, timestamps2, window=0.02):
    """
    Measure time-dependent timing offset between two telescopes.
    
    Finds coincident events within a large window and calculates the time offset dt=t1-t2
    for each match. Useful for diagnosing clock drift and synchronization issues between telescopes.
    
    Parameters
    ----------
    timestamps1 : array-like
        Timestamps for telescope 1. Can be float Unix seconds or datetime64.
    timestamps2 : array-like
        Timestamps for telescope 2. Can be float Unix seconds or datetime64.
    window : float, optional
        Time window for finding coincident events (default 0.02 seconds = 20 ms)
    
    Returns
    -------
    time_coinc1 : ndarray
        Coincident timestamps from telescope 1, preserving datetime64 inputs when provided.
    time_coinc2 : ndarray
        Coincident timestamps from telescope 2, preserving datetime64 inputs when provided.
    dt : ndarray
        Time differences (t1 - t2) for coincident events in seconds
    """
    timestamps1_arr = np.asarray(timestamps1)
    timestamps2_arr = np.asarray(timestamps2)

    timestamps1_is_datetime = np.issubdtype(timestamps1_arr.dtype, np.datetime64)
    timestamps2_is_datetime = np.issubdtype(timestamps2_arr.dtype, np.datetime64)

    if timestamps1_is_datetime:
        timestamps1_view = timestamps1_arr.astype("datetime64[ns]")
        t1 = timestamps1_view.astype("int64") * 1e-9
    else:
        timestamps1_view = timestamps1_arr
        t1 = np.asarray(timestamps1_arr, dtype=np.float64)

    if timestamps2_is_datetime:
        timestamps2_view = timestamps2_arr.astype("datetime64[ns]")
        t2 = timestamps2_view.astype("int64") * 1e-9
    else:
        timestamps2_view = timestamps2_arr
        t2 = np.asarray(timestamps2_arr, dtype=np.float64)

    # Find for each t1 the search range in t2
    left = np.searchsorted(t2, t1 - window, side="left")
    right = np.searchsorted(t2, t1 + window, side="right")

    idx1 = []
    idx2 = []

    for i, (l, r) in enumerate(zip(left, right)):
        if l < r:
            idx1.extend([i] * (r - l))
            idx2.extend(range(l, r))

    idx1 = np.asarray(idx1, dtype=np.int64)
    idx2 = np.asarray(idx2, dtype=np.int64)

    time_coinc1 = timestamps1_view[idx1] if timestamps1_is_datetime else t1[idx1]
    time_coinc2 = timestamps2_view[idx2] if timestamps2_is_datetime else t2[idx2]
    dt = t1[idx1] - t2[idx2]

    return time_coinc1, time_coinc2, dt


def correct_telescope_timing(timestamps1, timestamps2, window=0.02, bin_width=120):
    """
    Apply time-dependent offset correction to synchronize telescope timestamps.
    
    Measures the timing offset between telescopes in time bins and applies per-bin correction
    to all timestamps. After correction, timestamps1 should align with timestamp2 to within ~1ms,
    suitable for multi-telescope coincidence matching.
    
    Parameters
    ----------
    timestamps1 : array-like
        Unix timestamps for telescope 1 (will be corrected) - can be float or datetime64
    timestamps2 : array-like
        Unix timestamps for telescope 2 (reference) - can be float or datetime64
    window : float, optional
        Time window for finding coincident events (default 0.02 seconds)
    bin_width : float, optional
        Width of time bins for offset correction (default 120 seconds)
    
    Returns
    -------
    correction_result : dict
        Dictionary containing:
        - 'timestamps1_corr': Corrected timestamps for telescope 1 (ndarray)
        - 'time_coinc1': Pre-correction coincident timestamps (ndarray)
        - 'dt': Pre-correction timing differences (ndarray)
        - 'dt_median': Pre-correction median offset per bin (ndarray)
        - 'bin_edges': Bin edges used (ndarray)
        - 'time_coinc1_corr': Post-correction coincident timestamps (ndarray)
        - 'dt_corr': Post-correction timing differences (ndarray)
        - 'dt_median_corr': Post-correction median offset per bin (ndarray)
        - 'rms_before': RMS offset before correction (float)
        - 'rms_after': RMS offset after correction (float)
        - 'sigma_before': Std dev before correction (float)
        - 'sigma_after': Std dev after correction (float)
    """
    # Convert to float for calculations
    if np.issubdtype(np.asarray(timestamps1).dtype, np.datetime64):
        ts1_float = (np.asarray(timestamps1).astype("datetime64[ns]").astype("int64") * 1e-9)
    else:
        ts1_float = np.asarray(timestamps1, dtype=np.float64)
    
    if np.issubdtype(np.asarray(timestamps2).dtype, np.datetime64):
        ts2_float = (np.asarray(timestamps2).astype("datetime64[ns]").astype("int64") * 1e-9)
    else:
        ts2_float = np.asarray(timestamps2, dtype=np.float64)
    
    # Get timing differences in large window
    time_coinc1, _, dt = measure_telescope_timing_offset(ts1_float, ts2_float, window=window)
    
    logger.debug(f"  Coincidences found in ±{window}s window: {len(time_coinc1)} matches")
    if len(time_coinc1) == 0:
        logger.error(f"  NO COINCIDENCES FOUND! Timestamps1 range: {ts1_float.min():.3f}-{ts1_float.max():.3f}s, "
                    f"Timestamps2 range: {ts2_float.min():.3f}-{ts2_float.max():.3f}s")
        # Fallback: return unchanged timestamps if no matches found
        return {
            'timestamps1_corr': ts1_float,
            'time_coinc1': time_coinc1,
            'dt': dt,
            'dt_median': np.array([]),
            'bin_edges': np.array([]),
            'time_coinc1_corr': time_coinc1,
            'dt_corr': dt,
            'dt_median_corr': np.array([]),
            'rms_before': 0.0,
            'rms_after': 0.0,
            'sigma_before': 0.0,
            'sigma_after': 0.0
        }
    
    # Bin the timing differences (match correct_time: use actual min/max, fewer empty bins)
    bin_edges = np.arange(np.min(time_coinc1), np.max(time_coinc1) + bin_width + 1, bin_width)
    y, x = np.histogram(time_coinc1, bins=bin_edges)
    
    dt_median = np.zeros(len(x) - 1)
    for i in range(len(x) - 1):
        mask = (time_coinc1 >= x[i]) & (time_coinc1 < x[i + 1])
        if np.any(mask):
            dt_median[i] = np.median(dt[mask])
        else:
            dt_median[i] = np.nan
    
    # Compute RMS of offset before correction - ONLY include bins with real data (skip NaN)
    rms = np.sqrt(np.mean(np.square(dt_median[~np.isnan(dt_median)])))
    sigma = np.std(dt_median[~np.isnan(dt_median)])
    logger.info(f"Before correction - Offset RMS: {rms:.5f}s, std: {sigma:.5f}s")
    logger.debug(f"  Bins with real data: {np.sum(~np.isnan(dt_median))}/{len(dt_median)}")
    
    # Apply bin-wise correction to all timestamps1
    # Note: Timestamps in empty bins will become NaN, but only coincident-matched timestamps are used downstream
    t_corr = np.digitize(ts1_float, x) - 1
    t_corr = np.clip(t_corr, 0, len(dt_median) - 1)
    ts1_corrected = ts1_float - dt_median[t_corr]
    
    # Validate correction by recalculating offset
    time_coinc1_corr, _, dt_corr = measure_telescope_timing_offset(ts1_corrected, ts2_float, window=window)
    y_corr, x_corr = np.histogram(time_coinc1_corr, bins=bin_edges)
    
    dt_median_corr = np.zeros(len(x_corr) - 1)
    for i in range(len(x_corr) - 1):
        mask = (time_coinc1_corr >= x_corr[i]) & (time_coinc1_corr < x_corr[i + 1])
        if np.any(mask):
            dt_median_corr[i] = np.median(dt_corr[mask])
        else:
            dt_median_corr[i] = np.nan
    
    # RMS after correction - ONLY include bins with real data (skip NaN)
    rms_corr = np.sqrt(np.mean(np.square(dt_median_corr[~np.isnan(dt_median_corr)])))
    sigma_corr = np.std(dt_median_corr[~np.isnan(dt_median_corr)])
    logger.info(f"After correction - Offset RMS: {rms_corr:.5f}s, std: {sigma_corr:.5f}s")
    
    return {
        'timestamps1_corr': ts1_corrected,
        'time_coinc1': time_coinc1,
        'dt': dt,
        'dt_median': dt_median,
        'bin_edges': x,
        'time_coinc1_corr': time_coinc1_corr,
        'dt_corr': dt_corr,
        'dt_median_corr': dt_median_corr,
        'rms_before': rms,
        'rms_after': rms_corr,
        'sigma_before': sigma,
        'sigma_after': sigma_corr
    }
