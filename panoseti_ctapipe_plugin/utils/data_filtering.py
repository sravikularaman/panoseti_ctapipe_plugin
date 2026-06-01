"""
Data filtering functions for PANOSETI event pre-processing.

This module contains functions for:
- Packet loss filtering (incomplete data detection)
- Rate spike filtering (triggering anomalies removal)

Last modified: 22 April 2026
"""

import logging

import numpy as np

__all__ = [
    "filter_packet_loss",
    "filter_rate_spikes",
]

logger = logging.getLogger(__name__)


def filter_packet_loss(metadata, data=None, timestamps=None):
    """
    Filter out events with packet loss (pkt_num == 0 in any QUABO).

    When pkt_num == 0 for any QUABO, it indicates missing data for that
    QUABO. This function creates a mask to exclude such events.

    Parameters
    ----------
    metadata : dict
        Metadata dictionary from PFF file containing quabo_0, quabo_1, quabo_2, quabo_3 entries
    data : np.ndarray, optional
        Event data array. If provided, will be filtered and returned along with mask.
    timestamps : np.ndarray, optional
        Timestamp array corresponding to events. If omitted and the metadata
        contains ``tv_sec`` and ``tv_usec`` for all QUABOs, timestamps are
        derived using the original ``cut_pkt_loss_old``
        ``tv_sec + tv_usec * 1e-6`` logic and returned.

    Returns
    -------
    loss_mask : np.ndarray
        Boolean mask of events to keep (True = keep, False = packet loss)
    num_removed_due_to_pkt_loss : int
        Number of events removed due to packet loss
    data_filtered : np.ndarray, optional
        Filtered data array (only if data was provided)
    timestamps_filtered : np.ndarray, optional
        Filtered timestamp array (only if timestamps were provided)
    """
    # Get packet numbers from all 4 QUABOs
    pkt_num_0 = np.asarray(metadata["quabo_0"]["pkt_num"])
    pkt_num_1 = np.asarray(metadata["quabo_1"]["pkt_num"])
    pkt_num_2 = np.asarray(metadata["quabo_2"]["pkt_num"])
    pkt_num_3 = np.asarray(metadata["quabo_3"]["pkt_num"])

    # Create mask: keep events where ALL QUABOs have pkt_num != 0
    loss_mask = (pkt_num_0 != 0) & (pkt_num_1 != 0) & (pkt_num_2 != 0) & (pkt_num_3 != 0)

    num_removed_due_to_pkt_loss = np.sum(~loss_mask)
    pct = 100 * num_removed_due_to_pkt_loss / len(loss_mask) if len(loss_mask) > 0 else 0
    logger.info(
        f"Packet loss filter: removed {num_removed_due_to_pkt_loss} events ({pct:.2f}%)"
    )

    # Build return values based on what was provided
    return_values = [loss_mask, num_removed_due_to_pkt_loss]
    
    if data is not None:
        data_filtered = data[loss_mask]
        return_values.append(data_filtered)
    
    metadata_has_timestamps = all(
        "tv_sec" in metadata[f"quabo_{index}"] and "tv_usec" in metadata[f"quabo_{index}"]
        for index in range(4)
    )

    if timestamps is None and metadata_has_timestamps:
        timestamps_quabo = []
        for index in range(4):
            quabo = metadata[f"quabo_{index}"]
            timestamps_quabo.append(
                np.asarray(quabo["tv_sec"], dtype=np.float64)
                + np.asarray(quabo["tv_usec"], dtype=np.float64) * 1e-6
            )
        timestamps = np.min(np.asarray(timestamps_quabo), axis=0)

    if timestamps is not None:
        timestamps_filtered = timestamps[loss_mask]
        return_values.append(timestamps_filtered)

    return tuple(return_values) if len(return_values) > 2 else tuple(return_values)


def filter_rate_spikes(timestamps, bin_width=30, rate_threshold=2.0, data=None):
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
    data : np.ndarray, optional
        Event data array. If provided, will be filtered and returned.

    Returns
    -------
    spike_mask : np.ndarray
        Boolean mask of events to keep (True = keep, False = spike)
    num_removed_due_to_spike : int
        Number of events removed due to rate spikes
    data_filtered : np.ndarray, optional
        Filtered data array (only if data was provided)
    timestamps_filtered : np.ndarray
        Filtered timestamps array
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

    num_removed_due_to_spike = np.sum(~spike_mask)
    pct = 100 * num_removed_due_to_spike / len(timestamps_float) if len(timestamps_float) > 0 else 0
    logger.info(
        f"Rate spike filter (threshold={rate_threshold} Hz): "
        f"removed {num_removed_due_to_spike} events ({pct:.2f}%)"
    )

    # Build return values based on what was provided
    return_values = [spike_mask, num_removed_due_to_spike]
    
    if data is not None:
        data_filtered = data[spike_mask]
        return_values.append(data_filtered)

    timestamps_filtered = timestamps[spike_mask]
    return_values.append(timestamps_filtered)

    return tuple(return_values) if len(return_values) > 2 else tuple(return_values)
