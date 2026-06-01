"""
Visualization utilities for PANOSETI data analysis.

This module contains plotting functions for:
- Timing correction (offset measurement and correction validation)
- Data filtering (packet loss, rate spikes)
- Coincidence rate analysis

All plotting functions are designed to be called independently after data processing,
following ctapipe conventions of separating computation from visualization.

Last modified: 29 April 2026
"""

import logging
import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)

__all__ = [
    "plot_timing_offset",
    "plot_timing_correction_results",
    "plot_spike_rate_filter_comparison",
    "plot_packet_loss_filter_summary",
    "plot_trigger_rate_before_after_cuts",
]


# ==============================================================================
# TIMING CORRECTION VISUALIZATION
# ==============================================================================


def plot_timing_offset(time_coinc1, dt, title="Time offset between telescopes"):
    """
    Plot time offset between telescope pair before correction.
    
    Useful for diagnosing clock drift patterns and determining if correction is needed.
    
    Parameters
    ----------
    time_coinc1 : ndarray
        Unix timestamps of coincident events from telescope 1
    dt : ndarray
        Time differences (t1 - t2) for each coincidence
    title : str, optional
        Plot title (default: "Time offset between telescopes")
    
    Returns
    -------
    fig : matplotlib.figure.Figure or None
        Figure object if plotting succeeded, None otherwise
    """
    try:
        import matplotlib.pyplot as plt
        time_coinc_pd1 = pd.to_datetime(time_coinc1, unit='s', utc=True)
        fig = plt.figure(figsize=(12, 4))
        plt.scatter(time_coinc_pd1, dt, marker=".", s=5, alpha=0.5)
        plt.xlabel("Time (UTC)")
        plt.ylabel("Time offset [s]")
        plt.title(title)
        plt.grid(alpha=0.3)
        plt.tight_layout()
        return fig
    except ImportError:
        logger.warning("matplotlib not available for plotting")
        return None


def plot_timing_correction_results(time_coinc1, dt, dt_median, bin_edges, 
                                    time_coinc1_corr, dt_corr, dt_median_corr, 
                                    bin_edges_corr, rms_before, rms_after, 
                                    sigma_before, sigma_after):
    """
    Plot before/after timing correction comparison.
    
    Shows raw timing offset scatter points with median per-bin correction,
    then post-correction validation to verify alignment.
    
    Parameters
    ----------
    time_coinc1 : ndarray
        Pre-correction coincident timestamps
    dt : ndarray
        Pre-correction timing differences
    dt_median : ndarray
        Pre-correction median offset per bin
    bin_edges : ndarray
        Bin edges for pre-correction histogram
    time_coinc1_corr : ndarray
        Post-correction coincident timestamps
    dt_corr : ndarray
        Post-correction timing differences
    dt_median_corr : ndarray
        Post-correction median offset per bin
    bin_edges_corr : ndarray
        Bin edges for post-correction histogram
    rms_before : float
        RMS offset before correction (seconds)
    rms_after : float
        RMS offset after correction (seconds)
    sigma_before : float
        Std dev of offset before correction (seconds)
    sigma_after : float
        Std dev of offset after correction (seconds)
    
    Returns
    -------
    fig : matplotlib.figure.Figure or None
        Figure object if plotting succeeded, None otherwise
    """
    try:
        import matplotlib.pyplot as plt
        
        time_coinc_pd1 = pd.to_datetime(time_coinc1, unit='s', utc=True)
        time_coinc_pd1_corr = pd.to_datetime(time_coinc1_corr, unit='s', utc=True)
        x_pd = pd.to_datetime(bin_edges[:-1], unit='s', utc=True)
        
        fig, ax = plt.subplots(2, 1, figsize=(14, 8), sharex=True)
        
        # Before correction
        ax[0].scatter(time_coinc_pd1, dt, marker=".", s=5, alpha=0.5, label="Raw offsets")
        ax[0].step(x_pd, dt_median, where='mid', 
                   label=f"Median (RMS={rms_before:.5f}s, σ={sigma_before:.5f}s)", 
                   color="red", linewidth=2)
        ax[0].set_ylabel("Time offset [s]")
        ax[0].set_title("Before Timing Correction")
        ax[0].grid(alpha=0.3)
        ax[0].legend(loc='best')
        
        # After correction
        ax[1].scatter(time_coinc_pd1_corr, dt_corr, marker=".", s=5, alpha=0.5, 
                     label="Corrected offsets")
        ax[1].step(x_pd, dt_median_corr, where='mid', 
                   label=f"Median (RMS={rms_after:.5f}s, σ={sigma_after:.5f}s)", 
                   color="green", linewidth=2)
        ax[1].axhline(y=0, color='k', linestyle='--', linewidth=1, alpha=0.3)
        ax[1].set_xlabel("Time (UTC)")
        ax[1].set_ylabel("Time offset [s]")
        ax[1].set_title("After Timing Correction (Validation)")
        ax[1].grid(alpha=0.3)
        ax[1].legend(loc='best')
        
        fig.tight_layout()
        return fig
    except ImportError:
        logger.warning("matplotlib not available for plotting")
        return None


# ==============================================================================
# DATA FILTERING VISUALIZATION
# ==============================================================================


def plot_spike_rate_filter_comparison(timestamps_all, spike_mask, bin_width=30, 
                                      rate_threshold=2.0):
    """
    Plot trigger rate before and after rate spike filtering.
    
    Compares the raw trigger rate over time with spike-filtered rate, showing
    which bins exceed the threshold and were removed.
    
    Parameters
    ----------
    timestamps_all : ndarray
        Unix timestamps of ALL events (before filtering)
    spike_mask : ndarray
        Boolean mask of events to keep (True = keep, False = spike)
    bin_width : float, optional
        Time width for rate calculation in seconds (default 30)
    rate_threshold : float, optional
        Rate threshold in Hz (default 2.0 Hz)
    
    Returns
    -------
    fig : matplotlib.figure.Figure or None
        Figure object if plotting succeeded, None otherwise
    """
    try:
        import matplotlib.pyplot as plt
        
        # Convert datetime64 to float seconds if needed
        if np.issubdtype(timestamps_all.dtype, np.datetime64):
            timestamps_float = (
                timestamps_all.astype("datetime64[ns]").astype("int64") * 1e-9
            )
        else:
            timestamps_float = np.asarray(timestamps_all, dtype=np.float64)
        
        # Compute rate BEFORE filtering (all events)
        bins = np.arange(
            np.floor(timestamps_float.min()), 
            np.ceil(timestamps_float.max()) + bin_width, 
            bin_width
        )
        counts_all, _ = np.histogram(timestamps_float, bins=bins)
        rate_all = counts_all / bin_width
        time_bins = bins[:-1] + bin_width / 2.0
        time_bins_pd = pd.to_datetime(time_bins, unit='s', utc=True)
        
        # Compute rate AFTER filtering (only kept events)
        timestamps_filtered = timestamps_all[spike_mask]
        if np.issubdtype(timestamps_filtered.dtype, np.datetime64):
            timestamps_filtered_float = (
                timestamps_filtered.astype("datetime64[ns]").astype("int64") * 1e-9
            )
        else:
            timestamps_filtered_float = np.asarray(timestamps_filtered, dtype=np.float64)
        
        counts_filtered, _ = np.histogram(timestamps_filtered_float, bins=bins)
        rate_filtered = counts_filtered / bin_width
        
        # Identify spike bins
        spike_bins = rate_all > rate_threshold
        
        # Create figure
        fig, ax = plt.subplots(figsize=(14, 5))
        
        # Plot rates
        ax.step(time_bins_pd, rate_all, where='mid', label="Before filtering (all events)", 
                color='red', linewidth=2, alpha=0.7)
        ax.step(time_bins_pd, rate_filtered, where='mid', label="After filtering (kept events)", 
                color='green', linewidth=2, alpha=0.7)
        
        # Highlight spike bins
        spike_bin_times = time_bins_pd[spike_bins]
        spike_bin_rates = rate_all[spike_bins]
        ax.scatter(spike_bin_times, spike_bin_rates, color='red', s=100, marker='x', 
                   linewidth=3, label=f'Spike bins (>{rate_threshold} Hz)', zorder=5)
        
        # Threshold line
        ax.axhline(y=rate_threshold, color='gray', linestyle='--', linewidth=1.5, 
                   label=f'Threshold ({rate_threshold} Hz)', alpha=0.7)
        
        ax.set_xlabel("Time (UTC)")
        ax.set_ylabel("Trigger rate [Hz]")
        ax.set_title(f"Trigger Rate Spike Filter (bin_width={bin_width}s)")
        ax.legend(loc='best', fontsize=10)
        ax.grid(alpha=0.3)
        
        fig.tight_layout()
        return fig
    except ImportError:
        logger.warning("matplotlib not available for plotting")
        return None


def plot_packet_loss_filter_summary(timestamps_all, loss_mask, bin_width=30):
    """
    Plot packet loss filter impact: number of events removed per time bin.
    
    Shows which time intervals had packet loss and how many events were lost.
    
    Parameters
    ----------
    timestamps_all : ndarray
        Unix timestamps of ALL events (before filtering)
    loss_mask : ndarray
        Boolean mask of events to keep (True = keep, False = packet loss)
    bin_width : float, optional
        Time width for binning in seconds (default 30)
    
    Returns
    -------
    fig : matplotlib.figure.Figure or None
        Figure object if plotting succeeded, None otherwise
    """
    try:
        import matplotlib.pyplot as plt
        
        # Convert datetime64 to float seconds if needed
        if np.issubdtype(timestamps_all.dtype, np.datetime64):
            timestamps_float = (
                timestamps_all.astype("datetime64[ns]").astype("int64") * 1e-9
            )
        else:
            timestamps_float = np.asarray(timestamps_all, dtype=np.float64)
        
        # Bin data
        bins = np.arange(
            np.floor(timestamps_float.min()), 
            np.ceil(timestamps_float.max()) + bin_width, 
            bin_width
        )
        
        # Count events in each bin
        counts_all, _ = np.histogram(timestamps_float, bins=bins)
        
        # Count lost events in each bin
        lost_mask = ~loss_mask
        timestamps_lost = timestamps_all[lost_mask]
        if np.issubdtype(timestamps_lost.dtype, np.datetime64):
            timestamps_lost_float = (
                timestamps_lost.astype("datetime64[ns]").astype("int64") * 1e-9
            )
        else:
            timestamps_lost_float = np.asarray(timestamps_lost, dtype=np.float64)
        
        counts_lost, _ = np.histogram(timestamps_lost_float, bins=bins)
        counts_kept = counts_all - counts_lost
        
        time_bins = bins[:-1] + bin_width / 2.0
        time_bins_pd = pd.to_datetime(time_bins, unit='s', utc=True)
        
        # Create figure
        fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(14, 8), sharex=True)
        
        # Stacked bar chart: kept vs lost
        ax1.bar(time_bins_pd, counts_kept, width=pd.Timedelta(seconds=bin_width*0.8), 
                label='Kept events', color='green', alpha=0.7)
        ax1.bar(time_bins_pd, counts_lost, width=pd.Timedelta(seconds=bin_width*0.8), 
                bottom=counts_kept, label='Lost events (packet loss)', color='red', alpha=0.7)
        ax1.set_ylabel("Event count")
        ax1.set_title("Packet Loss Filter Impact (stacked)")
        ax1.legend(loc='best')
        ax1.grid(alpha=0.3, axis='y')
        
        # Loss rate
        loss_rate = np.zeros_like(counts_all, dtype=float)
        mask_nonzero = counts_all > 0
        loss_rate[mask_nonzero] = counts_lost[mask_nonzero] / counts_all[mask_nonzero] * 100
        
        colors = ['red' if lr > 0 else 'green' for lr in loss_rate]
        ax2.bar(time_bins_pd, loss_rate, width=pd.Timedelta(seconds=bin_width*0.8), 
                color=colors, alpha=0.7, label='Loss rate per bin')
        ax2.set_xlabel("Time (UTC)")
        ax2.set_ylabel("Packet loss rate [%]")
        ax2.set_title("Packet Loss Rate by Time Bin")
        ax2.grid(alpha=0.3, axis='y')
        
        fig.tight_layout()
        return fig
    except ImportError:
        logger.warning("matplotlib not available for plotting")
        return None


def plot_trigger_rate_before_after_cuts(timestamps_all, spike_mask, loss_mask, 
                                        bin_width=30, rate_threshold=2.0):
    """
    Comprehensive plot: trigger rate before and after all cuts (packet loss + spike rate).
    
    Shows how the trigger rate evolves through different filtering stages.
    
    Parameters
    ----------
    timestamps_all : ndarray
        Unix timestamps of ALL events (before any filtering)
    spike_mask : ndarray
        Boolean mask from rate spike filter (True = keep, False = spike)
    loss_mask : ndarray
        Boolean mask from packet loss filter (True = keep, False = loss)
    bin_width : float, optional
        Time width for rate calculation in seconds (default 30)
    rate_threshold : float, optional
        Rate threshold in Hz (default 2.0 Hz)
    
    Returns
    -------
    fig : matplotlib.figure.Figure or None
        Figure object if plotting succeeded, None otherwise
    """
    try:
        import matplotlib.pyplot as plt
        
        # Convert datetime64 to float seconds if needed
        if np.issubdtype(timestamps_all.dtype, np.datetime64):
            timestamps_float = (
                timestamps_all.astype("datetime64[ns]").astype("int64") * 1e-9
            )
        else:
            timestamps_float = np.asarray(timestamps_all, dtype=np.float64)
        
        # Bin data
        bins = np.arange(
            np.floor(timestamps_float.min()), 
            np.ceil(timestamps_float.max()) + bin_width, 
            bin_width
        )
        
        # Rate 1: All events (no filtering)
        counts_all, _ = np.histogram(timestamps_float, bins=bins)
        rate_all = counts_all / bin_width
        
        # Rate 2: After packet loss filter
        timestamps_after_loss = timestamps_all[loss_mask]
        if np.issubdtype(timestamps_after_loss.dtype, np.datetime64):
            timestamps_after_loss_float = (
                timestamps_after_loss.astype("datetime64[ns]").astype("int64") * 1e-9
            )
        else:
            timestamps_after_loss_float = np.asarray(timestamps_after_loss, dtype=np.float64)
        
        counts_after_loss, _ = np.histogram(timestamps_after_loss_float, bins=bins)
        rate_after_loss = counts_after_loss / bin_width
        
        # Rate 3: After both filters (packet loss AND spike rate)
        timestamps_final = timestamps_all[loss_mask & spike_mask]
        if np.issubdtype(timestamps_final.dtype, np.datetime64):
            timestamps_final_float = (
                timestamps_final.astype("datetime64[ns]").astype("int64") * 1e-9
            )
        else:
            timestamps_final_float = np.asarray(timestamps_final, dtype=np.float64)
        
        counts_final, _ = np.histogram(timestamps_final_float, bins=bins)
        rate_final = counts_final / bin_width
        
        time_bins = bins[:-1] + bin_width / 2.0
        time_bins_pd = pd.to_datetime(time_bins, unit='s', utc=True)
        
        # Create figure
        fig, ax = plt.subplots(figsize=(14, 6))
        
        # Plot all rates
        ax.step(time_bins_pd, rate_all, where='mid', label="No filtering", 
                color='red', linewidth=2.5, alpha=0.7, linestyle='-')
        ax.step(time_bins_pd, rate_after_loss, where='mid', label="After packet loss filter", 
                color='orange', linewidth=2, alpha=0.7, linestyle='--')
        ax.step(time_bins_pd, rate_final, where='mid', label="After packet loss + spike rate filters", 
                color='green', linewidth=2.5, alpha=0.7, linestyle='-')
        
        # Threshold line
        ax.axhline(y=rate_threshold, color='gray', linestyle='--', linewidth=1.5, 
                   label=f'Spike threshold ({rate_threshold} Hz)', alpha=0.7)
        
        # Statistics
        n_removed_by_loss = np.sum(~loss_mask)
        n_removed_by_spike = np.sum(~spike_mask)
        n_removed_by_both_not_loss = np.sum(loss_mask & ~spike_mask)
        n_removed_by_both_not_spike = np.sum(~loss_mask & spike_mask)
        n_removed_total = np.sum(~(loss_mask & spike_mask))
        
        n_all = len(timestamps_all)
        
        # Add text box with statistics
        stats_text = (
            f"Total events: {n_all}\n"
            f"Removed by packet loss: {n_removed_by_loss} ({100*n_removed_by_loss/n_all:.1f}%)\n"
            f"Removed by spike rate: {n_removed_by_spike} ({100*n_removed_by_spike/n_all:.1f}%)\n"
            f"Total removed: {n_removed_total} ({100*n_removed_total/n_all:.1f}%)\n"
            f"Kept events: {len(timestamps_final)} ({100*len(timestamps_final)/n_all:.1f}%)"
        )
        
        ax.text(0.02, 0.97, stats_text, transform=ax.transAxes, 
                verticalalignment='top', horizontalalignment='left',
                bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.8),
                fontsize=9, family='monospace')
        
        ax.set_xlabel("Time (UTC)")
        ax.set_ylabel("Trigger rate [Hz]")
        ax.set_title(f"Trigger Rate: Effect of Filtering Cuts (bin_width={bin_width}s)")
        ax.legend(loc='upper right', fontsize=10)
        ax.grid(alpha=0.3)
        
        fig.tight_layout()
        return fig
    except ImportError:
        logger.warning("matplotlib not available for plotting")
        return None
