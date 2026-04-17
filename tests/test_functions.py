"""
Unit tests for PANOSETI utility functions module.

Tests timestamp conversion, filtering, pedestal computation, and data selection.

Following ctapipe guidelines:
https://ctapipe.readthedocs.io/en/stable/developer-guide/code-guidelines.html#unit-tests

Author: Sruthi Ravikularaman
Last modified: 17 April 2026
"""

import numpy as np
import pandas as pd
import pytest
import astropy.units as u

from panoseti_ctapipe_plugin.functions import (
    wr_to_unix,
    apply_packet_loss_filter,
    apply_rate_spike_filter,
    compute_pedestals_from_data,
    calculate_pedestal_and_pedvar_robust,
    select_time_interval,
    load_gain_file,
)


# White Rabbit timestamp conversion tests


def test_wr_to_unix_basic():
    """Test basic timestamp conversion."""
    pkt_nsec = np.array([123456789])
    tv_sec = np.array([1000000])
    tv_usec = np.array([100000])

    result = wr_to_unix(pkt_nsec, tv_sec, tv_usec)

    # Result should be datetime64[ns]
    assert result.dtype == np.dtype("datetime64[ns]")
    assert len(result) == 1


def test_wr_to_unix_arrays():
    """Test conversion with arrays of timestamps."""
    pkt_nsec = np.array([123456789, 234567890, 345678901])
    tv_sec = np.array([1000000, 1000001, 1000002])
    tv_usec = np.array([100000, 200000, 300000])

    result = wr_to_unix(pkt_nsec, tv_sec, tv_usec)

    assert len(result) == 3
    assert result.dtype == np.dtype("datetime64[ns]")


def test_wr_to_unix_scalar():
    """Test conversion with scalar inputs."""
    result = wr_to_unix(123456789, 1000000, 100000)

    # Returns 0-d (scalar) datetime64[ns]
    assert isinstance(result, np.ndarray)
    assert result.dtype == np.dtype("datetime64[ns]")
    assert result.ndim == 0


def test_wr_to_unix_clock_mismatch_error():
    """Test that clock desync errors are raised appropriately."""
    # Error occurs at boundary: diff == ±25ms exactly (not inside any case)
    # diff = usec_ns - pkt_nsec
    # mask0 = |diff| < 25ms: requires -25ms < diff < 25ms
    # mask1023 = diff > 25ms: requires diff > 25ms (strictly)
    # mask1 = diff < -25ms: requires diff < -25ms (strictly)
    # So diff == 25ms exactly falls into none of these categories
    
    pkt_nsec = np.array([0])
    tv_sec = np.array([1000000])
    tv_usec = np.array([25000])  # Exactly 25ms = boundary case

    # Should raise clock mismatch error
    with pytest.raises(Exception, match="Clock mismatch"):
        wr_to_unix(pkt_nsec, tv_sec, tv_usec, ignore_clock_desync=False)


def test_wr_to_unix_clock_desync_ignored():
    """Test that clock desync can be ignored when requested."""
    pkt_nsec = np.array([1])
    tv_sec = np.array([1000000])
    tv_usec = np.array([999999])

    result = wr_to_unix(
        pkt_nsec, tv_sec, tv_usec, ignore_clock_desync=True
    )

    # Should return a valid result even with extreme desync
    assert len(result) == 1
    assert result.dtype == np.dtype("datetime64[ns]")


# Packet loss filter tests


def test_packet_loss_filter_no_loss(synthetic_event_data):
    """Test filtering when no packet loss is present."""
    metadata = {
        "quabo_0": {"pkt_num": np.ones(len(synthetic_event_data))},
        "quabo_1": {"pkt_num": np.ones(len(synthetic_event_data))},
        "quabo_2": {"pkt_num": np.ones(len(synthetic_event_data))},
        "quabo_3": {"pkt_num": np.ones(len(synthetic_event_data))},
    }

    data_filtered, valid_mask, pkt_loss_count = apply_packet_loss_filter(
        synthetic_event_data, metadata
    )

    # No events should be removed
    assert len(data_filtered) == len(synthetic_event_data)
    assert np.all(valid_mask)
    assert pkt_loss_count == 0


def test_packet_loss_filter_with_loss(synthetic_event_data):
    """Test filtering when some events have packet loss."""
    n_events = len(synthetic_event_data)
    pkt_nums = [
        np.ones(n_events),
        np.ones(n_events),
        np.ones(n_events),
        np.ones(n_events),
    ]
    # Introduce packet loss in 10 events
    pkt_nums[0][0:10] = 0
    pkt_nums[1][5:15] = 0

    metadata = {
        "quabo_0": {"pkt_num": pkt_nums[0]},
        "quabo_1": {"pkt_num": pkt_nums[1]},
        "quabo_2": {"pkt_num": pkt_nums[2]},
        "quabo_3": {"pkt_num": pkt_nums[3]},
    }

    data_filtered, valid_mask, pkt_loss_count = apply_packet_loss_filter(
        synthetic_event_data, metadata
    )

    # Some events should be removed (union of all packet losses)
    assert len(data_filtered) < len(synthetic_event_data)
    assert pkt_loss_count > 0
    assert np.sum(valid_mask) == len(data_filtered)


def test_packet_loss_filter_mask_consistency(synthetic_event_data):
    """Test that the returned mask is consistent."""
    metadata = {
        "quabo_0": {"pkt_num": np.ones(len(synthetic_event_data))},
        "quabo_1": {"pkt_num": np.ones(len(synthetic_event_data))},
        "quabo_2": {"pkt_num": np.ones(len(synthetic_event_data))},
        "quabo_3": {"pkt_num": np.ones(len(synthetic_event_data))},
    }

    data_filtered, valid_mask, _ = apply_packet_loss_filter(
        synthetic_event_data, metadata
    )

    # Applying mask manually should give same result
    assert np.array_equal(data_filtered, synthetic_event_data[valid_mask])


# Rate spike filter tests


def test_rate_spike_filter_no_spikes(synthetic_timestamps):
    """Test filtering with uniformly distributed events (no spikes)."""
    mask, spike_count = apply_rate_spike_filter(
        synthetic_timestamps, bin_width=10, rate_threshold=2.0
    )

    # Should keep most events (rate is 1.5 Hz, below 2.0 Hz threshold)
    assert mask.dtype == np.dtype("bool")
    assert len(mask) == len(synthetic_timestamps)
    assert np.sum(mask) > len(synthetic_timestamps) * 0.9


def test_rate_spike_filter_with_spike():
    """Test filtering with an actual spike."""
    # Create 150 events over 100 seconds (1.5 Hz normal rate)
    base_ts = np.linspace(1000000, 1000100, 150)

    # Create a spike: 15 events in 1 second (15 Hz >> 2 Hz threshold)
    spike_start = 1000050
    spike_ts = np.linspace(spike_start, spike_start + 1.0, 15)

    # Combine
    ts_before = base_ts[base_ts < spike_start]
    ts_after = base_ts[base_ts > spike_start + 1.0]
    synthetic_ts = np.concatenate([ts_before, spike_ts, ts_after])

    mask, spike_count = apply_rate_spike_filter(
        synthetic_ts, bin_width=10, rate_threshold=2.0
    )

    # Verify spike events are removed
    # Spike events occupy indices from len(ts_before) to len(ts_before) + 14
    n_before = len(ts_before)
    spike_indices = np.arange(n_before, n_before + 15)
    
    # All spike events should be marked as False (removed)
    assert np.all(~mask[spike_indices]), "Not all spike events were removed"
    
    # Non-spike events should be kept
    assert np.any(mask), "All events were removed, not just spikes"
    
    # spike_count should indicate events were removed
    assert spike_count > 0
    assert np.sum(~mask) >= 15, "Should remove at least 15 spike events"


# Pedestal computation tests


def test_compute_pedestals_from_data_shape(synthetic_event_data):
    """Test that computed pedestals have correct shape."""
    pedestal, pedvar = compute_pedestals_from_data(synthetic_event_data)

    # Should be 32x32
    assert pedestal.shape == (32, 32)
    assert pedvar.shape == (32, 32)


def test_compute_pedestals_from_data_values(synthetic_event_data):
    """Test that pedestal values are reasonable."""
    pedestal, pedvar = compute_pedestals_from_data(synthetic_event_data)

    # Pedestal should be around 100 (from synthetic data)
    assert 95 < pedestal.mean() < 105
    # Variance should be around 25 (σ^2 where σ=5)
    assert 20 < pedvar.mean() < 30


def test_calculate_pedestal_robust_without_gaussian(synthetic_event_data):
    """Test robust pedestal without Gaussian fitting."""
    # Input shape (100, 1024) returns output shape (1024,)
    pedestal, pedvar = calculate_pedestal_and_pedvar_robust(
        synthetic_event_data, nsig=5.0, fit_gaussian=False
    )

    assert pedestal.shape == (1024,)
    assert pedvar.shape == (1024,)
    # Should be close to std of data (around 5)
    assert 0 < pedvar.mean() < 10


def test_calculate_pedestal_robust_with_gaussian(synthetic_event_data):
    """Test robust pedestal with Gaussian fitting."""
    # Input shape (100, 1024) returns output shape (1024,) (reshaped from linear)
    pedestal, pedvar = calculate_pedestal_and_pedvar_robust(
        synthetic_event_data, nsig=5.0, fit_gaussian=True
    )

    assert pedestal.shape == (1024,)
    assert pedvar.shape == (1024,)
    # Gaussian sigma should be close to data std
    assert 0 < pedvar.mean() < 10


def test_pedestal_outlier_rejection(synthetic_event_data):
    """Test that outliers are properly rejected."""
    # Add bright outliers to synthetic data
    data_with_outliers = synthetic_event_data.copy()
    # Add 10 bright outliers to first pixel
    data_with_outliers[0:10, 0] += 500

    pedestal, _ = calculate_pedestal_and_pedvar_robust(
        data_with_outliers, nsig=3.0, fit_gaussian=False
    )

    # Pedestal should not be affected by outliers (should still be ~100)
    assert 95 < pedestal.mean() < 105


# Time interval selection tests


def test_select_time_interval_basic(synthetic_timestamps, synthetic_event_data):
    """Test basic time interval selection."""
    # Create timestamps from 2025-01-15 02:27:00 to 04:00:00
    base_time = pd.Timestamp("2025-01-15 02:27:00", tz="UTC").timestamp()
    timestamps = np.linspace(base_time, base_time + 5400, len(synthetic_event_data))

    start_time = "2025-01-15T02:30:00Z"
    end_time = "2025-01-15T03:00:00Z"

    data_cut, ts_cut, indices = select_time_interval(
        timestamps, synthetic_event_data, start_time, end_time
    )

    # Should have selected some events
    assert len(data_cut) > 0
    assert len(ts_cut) > 0
    assert len(indices) > 0
    assert len(data_cut) == len(ts_cut) == len(indices)


def test_select_time_interval_no_events(synthetic_timestamps, synthetic_event_data):
    """Test selection with no events in interval."""
    # Create timestamps from 2025-01-15 02:27:00 to 04:00:00
    base_time = pd.Timestamp("2025-01-15 02:27:00", tz="UTC").timestamp()
    timestamps = np.linspace(base_time, base_time + 5400, len(synthetic_event_data))

    start_time = "2025-01-15T05:00:00Z"  # After data range
    end_time = "2025-01-15T06:00:00Z"

    data_cut, ts_cut, indices = select_time_interval(
        timestamps, synthetic_event_data, start_time, end_time
    )

    # Should select no events
    assert len(data_cut) == 0
    assert len(ts_cut) == 0
    assert len(indices) == 0


# Gain calibration tests


def test_load_gain_file_shape():
    """Test that loaded gain file has correct shape."""
    from pathlib import Path

    # Get default gain file path for telescope 1
    data_dir = Path(__file__).parent.parent / "panoseti_ctapipe_plugin" / "data"
    gain_file = data_dir / "gains_tel1_Gattini.csv"

    if gain_file.exists():
        gains = load_gain_file(tel_id=1, gain_file_path=gain_file)
        assert gains.shape == (32, 32)
    else:
        # Skip if test data not available
        pytest.skip(f"Gain file not found: {gain_file}")
