"""
Utility functions for PANOSETI data processing.

This module contains functions for:
- Gain calibration loading
- Timestamp conversion (White Rabbit to Unix)
- Pre-filtering (packet loss, rate spikes)
- Pedestal computation with outlier removal
- Time interval selection

Author: Sruthi Ravikularaman
Last modified: 17 April 2026
"""

import logging
from pathlib import Path
from typing import Tuple

import numpy as np
import pandas as pd
from scipy.optimize import curve_fit

__all__ = [
    "load_gain_file",
    "wr_to_unix",
    "apply_packet_loss_filter",
    "apply_rate_spike_filter",
    "compute_pedestals_from_data",
    "calculate_pedestal_and_pedvar_robust",
    "select_time_interval",
]

logger = logging.getLogger(__name__)


# ==============================================================================
# GAIN CALIBRATION
# ==============================================================================


def load_gain_file(tel_id, gain_file_path):
    """
    Load per-pixel gain calibration from CSV file.

    Parameters
    ----------
    tel_id : int
        Telescope ID
    gain_file_path : str or Path
        Path to CSV file with 32x32 gain matrix

    Returns
    -------
    np.ndarray
        32x32 array of per-pixel gain values
    """
    try:
        df = pd.read_csv(gain_file_path, header=None)
        gains = df.values.astype(np.float32)
        if gains.shape != (32, 32):
            logger.warning(
                f"Gain file for tel {tel_id} has shape {gains.shape}, expected (32, 32)"
            )
        return gains
    except Exception as e:
        logger.error(f"Failed to load gain file for tel {tel_id} ({gain_file_path}): {e}")
        raise

# ==============================================================================
# TIMESTAMP AND DATA FILTERING
# ==============================================================================


def wr_to_unix(pkt_nsec, tv_sec, tv_usec, ignore_clock_desync=False):
    """
    Convert White Rabbit timestamps to unix time (datetime64[ns]).

    Handles clock desynchronization between White Rabbit and Unix clocks
    using heuristics to determine the correct second boundary.

    Parameters
    ----------
    pkt_nsec : int or array-like
        Packet nanoseconds
    tv_sec : int or array-like
        Unix seconds
    tv_usec : int or array-like
        Unix microseconds
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

    # Convert tv_usec to ns
    usec_ns = tv_usec * 1000

    # Difference in ns
    diff = usec_ns - pkt_nsec

    # 25ms threshold in ns
    TH = 25_000_000

    # Determine which case applies
    mask0 = np.abs(diff) < TH  # Normal case: same second
    mask1023 = diff > TH  # tv_usec much larger: second +1
    mask1 = diff < -TH  # pkt_nsec much larger: second -1

    # Preallocate output as ns integer
    out_ns = np.empty_like(tv_sec, dtype=np.int64)

    # Normal case: same second
    out_ns[mask0] = (
        tv_sec[mask0] * 1_000_000_000 + pkt_nsec[mask0] + usec_ns[mask0]
    )

    # Case 1: pkt_nsec much larger → second -1
    out_ns[mask1] = (
        (tv_sec[mask1] - 1) * 1_000_000_000 + pkt_nsec[mask1] + usec_ns[mask1]
    )

    # Case 1023: tv_usec much larger → second +1
    out_ns[mask1023] = (
        (tv_sec[mask1023] + 1) * 1_000_000_000 + pkt_nsec[mask1023] + usec_ns[mask1023]
    )

    # Handle bad cases
    mask_bad = ~(mask0 | mask1 | mask1023)

    if np.any(mask_bad):
        if ignore_clock_desync:
            out_ns[mask_bad] = (
                tv_sec[mask_bad] * 1_000_000_000
                + pkt_nsec[mask_bad]
                + usec_ns[mask_bad]
            )
        else:
            i = np.flatnonzero(mask_bad)[0]
            raise Exception(
                f"Clock mismatch: tv_sec={tv_sec[i]} tv_usec={tv_usec[i]} "
                f"pkt_nsec={pkt_nsec[i]} diff={diff[i]}"
            )

    # Convert to datetime64[ns]
    return out_ns.astype("datetime64[ns]")


def apply_packet_loss_filter(data, metadata):
    """
    Remove events with packet loss from any QUABO.

    If any QUABO has pkt_num == 0 for an event, that event is dropped.

    Parameters
    ----------
    data : np.ndarray
        Event data array
    metadata : dict
        Metadata dictionary with QUABO information

    Returns
    -------
    data_filtered : np.ndarray
        Data with packet-loss events removed
    valid_mask : np.ndarray
        Boolean mask of valid events
    pkt_loss_count : int
        Number of events removed
    """
    pkt_num_0 = metadata["quabo_0"]["pkt_num"]
    pkt_num_1 = metadata["quabo_1"]["pkt_num"]
    pkt_num_2 = metadata["quabo_2"]["pkt_num"]
    pkt_num_3 = metadata["quabo_3"]["pkt_num"]

    # Valid events: all QUABOs have pkt_num != 0
    valid_mask = (
        (pkt_num_0 != 0) & (pkt_num_1 != 0) & (pkt_num_2 != 0) & (pkt_num_3 != 0)
    )

    data_filtered = data[valid_mask]
    pkt_loss_count = len(data) - len(data_filtered)

    pct = 100 * pkt_loss_count / len(data) if len(data) > 0 else 0
    logger.info(f"Packet loss filter: removed {pkt_loss_count} events ({pct:.2f}%)")

    return data_filtered, valid_mask, pkt_loss_count


def apply_rate_spike_filter(timestamps, bin_width=30, rate_threshold=2.0):
    """
    Filter out trigger rate spikes (e.g., from planes, cosmic rays).

    Divides data into time bins and removes events in bins exceeding
    the rate threshold.

    Parameters
    ----------
    timestamps : np.ndarray
        Unix timestamps (float or datetime64)
    bin_width : float
        Time width for rate calculation in seconds
    rate_threshold : float
        Rate threshold in Hz; bins exceeding this are removed

    Returns
    -------
    spike_mask : np.ndarray
        Boolean mask of events to keep (True = keep, False = spike)
    spike_count : int
        Number of events removed
    """
    # Convert datetime64 to float seconds if needed
    if np.issubdtype(timestamps.dtype, np.datetime64):
        timestamps_float = (
            timestamps.astype("datetime64[ns]").astype("int64") * 1e-9
        )
    else:
        timestamps_float = np.asarray(timestamps, dtype=np.float64)

    # Create bins and compute rate
    bins = np.arange(
        timestamps_float.min(), timestamps_float.max() + bin_width, bin_width
    )
    counts, _ = np.histogram(timestamps_float, bins=bins)
    rate = counts / bin_width  # Hz

    # Find bad bins (exceeding threshold)
    bad_bins = rate > rate_threshold

    # Assign each event to a bin
    bin_indices = np.digitize(timestamps_float, bins) - 1

    # Clip indices to valid range [0, len(bad_bins)-1]
    bin_indices = np.clip(bin_indices, 0, len(bad_bins) - 1)

    # Create mask: keep events NOT in bad bins
    spike_mask = ~bad_bins[bin_indices]

    spike_count = np.sum(~spike_mask)
    pct = 100 * spike_count / len(timestamps_float) if len(timestamps_float) > 0 else 0
    logger.info(
        f"Rate spike filter (threshold={rate_threshold} Hz): "
        f"removed {spike_count} events ({pct:.2f}%)"
    )

    return spike_mask, spike_count


# ==============================================================================
# PEDESTAL COMPUTATION
# ==============================================================================


def compute_pedestals_from_data(data_array):
    """
    Compute per-pixel pedestal (mean) and variance from a set of events.

    Parameters
    ----------
    data_array : np.ndarray
        Array of shape (n_events, 1024) containing raw pulse heights

    Returns
    -------
    pedestal : np.ndarray
        Shape (32, 32) per-pixel mean values
    pedvar : np.ndarray
        Shape (32, 32) per-pixel variance values
    """
    # Reshape all events to (n_events, 32, 32) and compute mean and variance
    n_events = len(data_array)
    images = np.array(
        [np.array(event, dtype=np.float32).reshape((32, 32)) for event in data_array]
    )
    pedestal = np.mean(images, axis=0)
    pedvar = np.var(images, axis=0)
    logger.info(
        f"Computed pedestal from {n_events} events: "
        f"mean={pedestal.mean():.2f}, std={pedestal.std():.2f}"
    )
    logger.info(
        f"Pedestal variance: mean={pedvar.mean():.2f}, std={pedvar.std():.2f}"
    )
    return pedestal, pedvar


def _gaussian(x, A, mu, sigma):
    """Gaussian function for pedestal variance fitting."""
    return A * np.exp(-((x - mu) ** 2) / (2 * sigma**2))


def calculate_pedestal_and_pedvar_robust(
    data, nsig=5.0, fit_gaussian=True
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Calculate pedestal and pedestal variance with outlier removal.

    Returns pedestal (mean) and pedvar (std or Gaussian-fitted sigma) per pixel.
    Bright outliers (> nsig * sigma above mean) are masked and excluded.

    Parameters
    ----------
    data : np.ndarray
        Data frames in format (n_frames, 1024) or (n_frames, 32, 32)
    nsig : float
        Number of sigmas above mean to define outlier threshold (default = 5.0)
    fit_gaussian : bool
        If True, fit Gaussian to masked data for pedvar.
        If False, use std of masked data.

    Returns
    -------
    mean_pixels : np.ndarray
        Shape (1024,) or (32, 32) pedestal (mean) per pixel
    sigma_pixels : np.ndarray
        Shape (1024,) or (32, 32) pedestal variance (Gaussian sigma or std)
    """
    # Flatten to (n_frames, n_pixels) if needed
    original_shape = data.shape
    if len(data.shape) == 3:
        n_frames, nx, ny = data.shape
        data_flat = data.reshape((n_frames, nx * ny))
    else:
        data_flat = data
        n_frames, n_pixels = data_flat.shape
        nx, ny = 32, 32

    n_pixels = data_flat.shape[1]

    # Initial estimates (robust to outliers)
    mean_pixels_initial = np.nanmean(data_flat, axis=0)
    sigma_pixels_initial = np.abs(np.nanstd(data_flat, axis=0))

    # Mask outliers: keep only data < mean + nsig*sigma
    threshold = mean_pixels_initial + nsig * sigma_pixels_initial
    data_masked = np.where(data_flat < threshold[None, :], data_flat, np.nan)

    # Recompute mean on masked data
    mean_pixels = np.nanmean(data_masked, axis=0)

    # If no Gaussian fit, return std of masked data
    if not fit_gaussian:
        sigma_pixels = np.abs(np.nanstd(data_masked, axis=0))
        return mean_pixels.reshape(original_shape[1:]), sigma_pixels.reshape(
            original_shape[1:]
        )

    # Gaussian fitting per pixel
    sigma_pixels = np.zeros(n_pixels)

    for i in range(n_pixels):
        x = data_masked[:, i]
        x_clean = x[np.isfinite(x)]  # Remove NaN values

        mu0 = mean_pixels[i]
        sigma0 = sigma_pixels_initial[i]

        if x_clean.size < 5:
            sigma_pixels[i] = sigma0
            continue

        # Histogram of clean data
        hmin, hmax, hbins = -500, 500, 1000
        hist, edges = np.histogram(x_clean, bins=hbins, range=(hmin, hmax))
        centers = 0.5 * (edges[1:] + edges[:-1])

        try:
            p0 = [hist.max(), mu0, sigma0]
            popt, _ = curve_fit(_gaussian, centers, hist, p0=p0, maxfev=2000)
            sigma_pixels[i] = abs(popt[2])
        except RuntimeError:
            logger.debug(f"Gaussian fit failed for pixel {i}, using initial sigma")
            sigma_pixels[i] = sigma0

    # Reshape back to original spatial dimensions
    return mean_pixels.reshape(original_shape[1:]), sigma_pixels.reshape(
        original_shape[1:]
    )


def select_time_interval(timestamps: np.ndarray, data: np.ndarray, start_time, end_time) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Select data within a time interval.

    Parameters
    ----------
    timestamps : np.ndarray
        1D array of unix timestamps (seconds)
    data : np.ndarray
        Data frames in format (n_frames, n_pixels)
    start_time : str or pd.Timestamp
        Start time in format 'YYYY-MM-DDTHH:MM:SSZ' or Pandas Timestamp
    end_time : str or pd.Timestamp
        End time in format 'YYYY-MM-DDTHH:MM:SSZ' or Pandas Timestamp

    Returns
    -------
    data_cut : np.ndarray
        Data within the interval
    timestamps_cut : np.ndarray
        Timestamps within the interval
    indices : np.ndarray
        Indices of selected events in original arrays
    """
    # Convert timestamps to pandas DatetimeIndex
    timestamps_df = pd.to_datetime(timestamps, unit="s", utc=True)

    # Convert time strings to Timestamp if needed
    if isinstance(start_time, str):
        start_time = pd.to_datetime(start_time, utc=True)
    if isinstance(end_time, str):
        end_time = pd.to_datetime(end_time, utc=True)

    # Find index range
    start_idx = timestamps_df.searchsorted(start_time, side="left")
    end_idx = timestamps_df.searchsorted(end_time, side="right")

    indices = np.arange(start_idx, end_idx)
    data_cut = data[start_idx:end_idx]
    timestamps_cut = timestamps[start_idx:end_idx]

    logger.info(f"Selected {len(data_cut)} events from {start_time} to {end_time}")

    return data_cut, timestamps_cut, indices
