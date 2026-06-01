"""
Multi-telescope coincidence matching functions for PANOSETI data processing.

This module contains functions for:
- Two-telescope coincidence matching
- Multi-telescope (3+) hierarchical coincidence matching
- Coincidence rate calculation
- Flexible coincidence event assembly

Last modified: 22 April 2026
"""

import logging

import numpy as np

__all__ = [
    "find_two_telescope_coincident_events",
    "find_multi_telescope_coincident_events",
    "calculate_coincidence_rate",
    "match_coincident_events",
]

logger = logging.getLogger(__name__)


def find_two_telescope_coincident_events(timestamps1, data1, timestamps2, data2, window=0.001):
    """
    Find coincident events between 2 telescopes within a tight time window.
    
    Identifies events from two telescopes that occur within the specified time window.
    Assumes timestamps have been corrected by correct_telescope_timing() before calling.
    Uses efficient binary search for fast matching of thousands of events.
    
    Parameters
    ----------
    timestamps1 : array-like
        Unix timestamps for telescope 1 (should be corrected)
    data1 : array-like
        Event data for telescope 1
    timestamps2 : array-like
        Unix timestamps for telescope 2 (should be corrected)
    data2 : array-like
        Event data for telescope 2
    window : float, optional
        Time window for matching (default 0.001 seconds = 1 ms)
    
    Returns
    -------
    time_coinc1 : ndarray
        Coincident timestamps from telescope 1
    data_coinc1 : ndarray
        Coincident data from telescope 1
    time_coinc2 : ndarray
        Coincident timestamps from telescope 2
    data_coinc2 : ndarray
        Coincident data from telescope 2
    """
    t1 = np.asarray(timestamps1, dtype=np.float64)
    t2 = np.asarray(timestamps2, dtype=np.float64)
    d1 = np.asarray(data1)
    d2 = np.asarray(data2)
    
    logger.info(f"Matching coincidences: Tel1 has {len(t1)} events, Tel2 has {len(t2)} events")
    
    # Candidate ranges in t2 for each t1
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
    
    time_coinc1 = t1[idx1]
    time_coinc2 = t2[idx2]
    data_coinc1 = d1[idx1]
    data_coinc2 = d2[idx2]
    
    ncoinc = len(time_coinc1)
    p1 = (ncoinc * 100.0 / len(t1)) if len(t1) > 0 else 0.0
    p2 = (ncoinc * 100.0 / len(t2)) if len(t2) > 0 else 0.0
    logger.info(f"Found {ncoinc} coincidences within {window}s window "
                f"({p1:.1f}% of Tel1, {p2:.1f}% of Tel2)")
    
    return time_coinc1, data_coinc1, time_coinc2, data_coinc2


def find_multi_telescope_coincident_events(timestamps1, data1, timestamps2, data2, timestamps3, data3, window=0.001):
    """
    Find coincident events across 3 telescopes within a tight time window.
    
    Identifies events from three telescopes that occur within the specified time window.
    Uses a hierarchical 2-step matching process: first finds 2-telescope coincidences (T1 & T2),
    then matches T3 to each pair. Assumes timestamps have been corrected beforehand.
    
    Parameters
    ----------
    timestamps1, timestamps2, timestamps3 : array-like
        Unix timestamps for telescopes 1, 2, 3 (should be corrected)
    data1, data2, data3 : array-like
        Event data for telescopes 1, 2, 3
    window : float, optional
        Time window for matching (default 0.001 seconds = 1 ms)
    
    Returns
    -------
    time1, data1_coinc, time2, data2_coinc, time3, data3_coinc : tuple
        Coincident timestamps and data for all 3 telescopes
    """
    t1 = np.asarray(timestamps1, dtype=np.float64)
    t2 = np.asarray(timestamps2, dtype=np.float64)
    t3 = np.asarray(timestamps3, dtype=np.float64)
    d1 = np.asarray(data1)
    d2 = np.asarray(data2)
    d3 = np.asarray(data3)
    
    logger.info(f"3-telescope matching: T1={len(t1)}, T2={len(t2)}, T3={len(t3)} events")
    
    # Step 1: Find all (i,j) pairs between T1 and T2 within ±window
    left2 = np.searchsorted(t2, t1 - window, side="left")
    right2 = np.searchsorted(t2, t1 + window, side="right")
    
    idx1_pairs = []
    idx2_pairs = []
    for i, (l, r) in enumerate(zip(left2, right2)):
        if l < r:
            idx1_pairs.extend([i] * (r - l))
            idx2_pairs.extend(range(l, r))
    
    idx1_pairs = np.asarray(idx1_pairs, dtype=np.int64)
    idx2_pairs = np.asarray(idx2_pairs, dtype=np.int64)
    
    if idx1_pairs.size == 0:
        logger.warning(f"No 2-telescope coincidences found between T1 and T2")
        empty_array = np.asarray([], dtype=float)
        return empty_array, empty_array, empty_array, empty_array, empty_array, empty_array
    
    # Representative time for each pair
    t_pair = 0.5 * (t1[idx1_pairs] + t2[idx2_pairs])
    
    # Step 2: For each (i,j) pair, find all k in T3 within ±window around t_pair
    left3 = np.searchsorted(t3, t_pair - window, side="left")
    right3 = np.searchsorted(t3, t_pair + window, side="right")
    
    idx1 = []
    idx2 = []
    idx3 = []
    
    for p, (l, r) in enumerate(zip(left3, right3)):
        if l < r:
            idx1.extend([idx1_pairs[p]] * (r - l))
            idx2.extend([idx2_pairs[p]] * (r - l))
            idx3.extend(range(l, r))
    
    idx1 = np.asarray(idx1, dtype=np.int64)
    idx2 = np.asarray(idx2, dtype=np.int64)
    idx3 = np.asarray(idx3, dtype=np.int64)
    
    time1 = t1[idx1]
    time2 = t2[idx2]
    time3 = t3[idx3]
    
    data1_coinc = d1[idx1]
    data2_coinc = d2[idx2]
    data3_coinc = d3[idx3]
    
    n = len(time1)
    p1 = (n * 100.0 / len(t1)) if len(t1) > 0 else 0.0
    p2 = (n * 100.0 / len(t2)) if len(t2) > 0 else 0.0
    p3 = (n * 100.0 / len(t3)) if len(t3) > 0 else 0.0
    logger.info(f"Found {n} 3-telescope coincidences ({p1:.1f}% of T1, {p2:.1f}% of T2, {p3:.1f}% of T3)")
    
    return time1, data1_coinc, time2, data2_coinc, time3, data3_coinc


def calculate_coincidence_rate(coinc_timestamps, bin_width=240):
    """
    Calculate time-binned coincidence rate from event timestamps.
    
    Histograms coincident timestamps into uniform time bins and computes the rate (Hz)
    for each bin. Useful for characterizing observation conditions and detector stability.
    
    Parameters
    ----------
    coinc_timestamps : array-like
        Unix timestamps of coincident events
    bin_width : float, optional
        Width of time bins in seconds (default 240 seconds = 4 minutes)
    
    Returns
    -------
    time_rate : ndarray
        Unix timestamps at center of each bin
    rate : ndarray
        Coincidence rate in Hz for each bin
    """
    coinc_ts = np.asarray(coinc_timestamps, dtype=np.float64)
    
    bin_edges = np.arange(np.floor(coinc_ts.min()), np.ceil(coinc_ts.max()) + bin_width, bin_width)
    y, x = np.histogram(coinc_ts, bins=bin_edges)
    
    time_rate = x[:-1] + bin_width / 2.0
    rate = y / bin_width
    
    return time_rate, rate


def match_coincident_events(timestamps_dict, data_dict=None, time_window=0.001):
    """
    Match coincident events across multiple telescopes hierarchically.

    Follows the approach from legacy Coincidences.py: builds pairs incrementally,
    allowing multiple matches at higher telescope levels for the same lower-level pair.
    For example, a single (T1,T2) pair can match multiple T3 events, creating 
    multiple triplets from that pair.

    Parameters
    ----------
    timestamps_dict : dict
        Dictionary mapping tel_id to numpy array of timestamps (datetime64[ns] or float)
    data_dict : dict, optional
        Dictionary mapping tel_id to numpy array of event data.
        If provided, will be grouped according to coincidences.
    time_window : float, optional
        Time window in seconds for matching coincident events (default: 0.001 = 1ms)

    Returns
    -------
    coincidences : list of dict
        Each dict represents one coincidence with:
        - 'event_time': minimum timestamp across matched telescopes (datetime64 or float)
        - 'tel_ids': list of telescope IDs with this event (sorted)
        - 'indices': dict mapping tel_id to index in original data
        - 'data': dict mapping tel_id to event data (only if data_dict provided)
    """
    # Debug: print raw input values
    for tel_id, ts in timestamps_dict.items():
        logger.debug(f"  Input timestamps[{tel_id}]: len={len(ts) if hasattr(ts, '__len__') else 'N/A'}, dtype={ts.dtype if hasattr(ts, 'dtype') else 'N/A'}")
    
    # Convert all timestamps to float seconds for comparison
    timestamps_float = {}
    for tel_id, ts in timestamps_dict.items():
        if np.issubdtype(ts.dtype, np.datetime64):
            timestamps_float[tel_id] = (
                ts.astype("datetime64[ns]").astype("int64") * 1e-9
            )
        else:
            timestamps_float[tel_id] = np.asarray(ts, dtype=np.float64)
    
    # Debug: print after conversion
    for tel_id, ts in timestamps_float.items():
        valid_count = np.sum(~np.isnan(ts))
        logger.debug(f"  After conversion[{tel_id}]: {valid_count} valid values")

    # Sort telescopes by ID for consistent ordering
    tel_ids = sorted(timestamps_dict.keys())

    if len(tel_ids) == 1:
        # Single telescope: every event is a coincidence with itself
        coincidences = []
        for i in range(len(timestamps_float[tel_ids[0]])):
            result = {
                'event_time': timestamps_dict[tel_ids[0]][i],
                'tel_ids': tel_ids,
                'indices': {tel_ids[0]: i},
            }
            if data_dict is not None:
                result['data'] = {tel_ids[0]: data_dict[tel_ids[0]][i]}
            coincidences.append(result)
        return coincidences

    # ===== HIERARCHICAL MATCHING (matching Coincidences.py) =====
    # Start with pairs between first two telescopes
    t1_id = tel_ids[0]
    t2_id = tel_ids[1]
    t1_ts = timestamps_float[t1_id]
    t2_ts = timestamps_float[t2_id]
    
    logger.debug(f"T1 ({t1_id}): {len(t1_ts)} events, range {t1_ts.min():.6f}-{t1_ts.max():.6f}s")
    logger.debug(f"T2 ({t2_id}): {len(t2_ts)} events, range {t2_ts.min():.6f}-{t2_ts.max():.6f}s")
    logger.debug(f"Time window: ±{time_window}s = ±{time_window*1e3:.1f}ms")

    # Find all (i,j) pairs where T1[i] and T2[j] are within time_window
    left = np.searchsorted(t2_ts, t1_ts - time_window, side="left")
    right = np.searchsorted(t2_ts, t1_ts + time_window, side="right")

    pairs = []
    for i, (l, r) in enumerate(zip(left, right)):
        if l < r:  # Found at least one match
            for j in range(l, r):
                pairs.append({
                    'tel_ids': [t1_id, t2_id],
                    'indices': {t1_id: i, t2_id: j},
                    'timestamps': {
                        t1_id: timestamps_dict[t1_id][i],
                        t2_id: timestamps_dict[t2_id][j],
                    },
                    'timestamps_float': {
                        t1_id: t1_ts[i],
                        t2_id: t2_ts[j],
                    },
                })
    
    logger.debug(f"Found {len(pairs)} initial (T1,T2) pairs")

    # Store all coincidences at each level (2-tel, 3-tel, 4-tel, etc.)
    # Users may want 2-telescope or 3-telescope events, not just complete 4-tel
    all_coincidences = []
    
    # Add all 2-telescope pairs as valid coincidences
    for pair in pairs:
        all_coincidences.append(pair.copy())

    # Extend pairs with each additional telescope hierarchically
    # But KEEP the previous-level pairs even if they don't extend
    for next_tel_id in tel_ids[2:]:
        next_ts = timestamps_float[next_tel_id]
        extended_pairs = []

        for pair in pairs:
            # Use midpoint of current pair as reference time for matching next telescope
            # Use float timestamps for arithmetic
            ts_vals_float = list(pair['timestamps_float'].values())
            ref_time = 0.5 * (min(ts_vals_float) + max(ts_vals_float))

            # Find ALL events in next_tel within time_window of ref_time
            left = np.searchsorted(next_ts, ref_time - time_window, side="left")
            right = np.searchsorted(next_ts, ref_time + time_window, side="right")

            # Each matching event creates a new coincidence extending the pair
            if left < right:
                for k in range(left, right):
                    extended_pair = {
                        'tel_ids': pair['tel_ids'] + [next_tel_id],
                        'indices': pair['indices'].copy(),
                        'timestamps': pair['timestamps'].copy(),
                        'timestamps_float': pair['timestamps_float'].copy(),
                    }
                    extended_pair['indices'][next_tel_id] = k
                    extended_pair['timestamps'][next_tel_id] = timestamps_dict[next_tel_id][k]
                    extended_pair['timestamps_float'][next_tel_id] = next_ts[k]
                    extended_pairs.append(extended_pair)

        # Add all extended pairs at this level as valid coincidences
        for extended_pair in extended_pairs:
            all_coincidences.append(extended_pair.copy())
        
        # Continue extension with the successfully extended pairs only
        pairs = extended_pairs
        logger.debug(f"Found {len(extended_pairs)} extended coincidences with {next_tel_id}")

    # Convert all coincidences to final result format
    coincidences = []
    for pair in all_coincidences:
        # Find earliest timestamp using float values (to handle mixed datetime64/float)
        # Then use the actual timestamp from the original dict
        ts_float_vals = list(pair['timestamps_float'].values())
        min_ts_float = min(ts_float_vals)
        
        # Find which tel_id has this min timestamp and use its original format
        for tel_id in pair['tel_ids']:
            if pair['timestamps_float'][tel_id] == min_ts_float:
                event_time = pair['timestamps'][tel_id]
                break
        else:
            # Fallback: just use the first timestamp
            event_time = pair['timestamps'][pair['tel_ids'][0]]
        
        result = {
            'event_time': event_time,
            'tel_ids': sorted(pair['tel_ids']),
            'indices': pair['indices'],
        }
        if data_dict is not None:
            result['data'] = {}
            for tel_id in result['tel_ids']:
                result['data'][tel_id] = data_dict[tel_id][pair['indices'][tel_id]]
        coincidences.append(result)

    logger.info(
        f"Coincidence matching (hierarchical): found {len(coincidences)} events "
        f"from {len(tel_ids)} telescopes within {time_window}s window"
    )

    return coincidences
