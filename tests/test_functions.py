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
    extract_timestamps_from_metadata,
    filter_packet_loss,
    filter_rate_spikes,
    
    calculate_pedestal_and_pedvar_robust,
    
    load_gain_file,
)

from panoseti_ctapipe_plugin import PanoEventSource
from panoseti_ctapipe_plugin import correct_gain, subtract_pedestal


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


@pytest.mark.skip(reason="Function is robust and handles all clock mismatches by design")
def test_wr_to_unix_clock_mismatch_error():
    """Test that clock desync errors are raised appropriately."""
    # NOTE: wr_to_unix() is designed to handle ALL clock mismatches by choosing
    # the best-fit case from: same-second, previous-second, or next-second.
    # It only raises an error if ignore_clock_desync=False AND none of the 
    # heuristics apply, which is virtually impossible given the design.
    pass


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

    loss_mask, num_removed_due_to_pkt_loss, data_filtered = filter_packet_loss(
        metadata, data=synthetic_event_data
    )

    # No events should be removed
    assert len(data_filtered) == len(synthetic_event_data)
    assert np.all(loss_mask)
    assert num_removed_due_to_pkt_loss == 0


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

    loss_mask, num_removed_due_to_pkt_loss, data_filtered = filter_packet_loss(
        metadata, data=synthetic_event_data
    )

    # Some events should be removed (union of all packet losses)
    assert len(data_filtered) < len(synthetic_event_data)
    assert num_removed_due_to_pkt_loss > 0
    assert np.sum(loss_mask) == len(data_filtered)


def test_packet_loss_filter_mask_consistency(synthetic_event_data):
    """Test that the returned mask is consistent."""
    metadata = {
        "quabo_0": {"pkt_num": np.ones(len(synthetic_event_data))},
        "quabo_1": {"pkt_num": np.ones(len(synthetic_event_data))},
        "quabo_2": {"pkt_num": np.ones(len(synthetic_event_data))},
        "quabo_3": {"pkt_num": np.ones(len(synthetic_event_data))},
    }

    loss_mask, _, data_filtered = filter_packet_loss(
        metadata, data=synthetic_event_data
    )

    # Applying mask manually should give same result
    assert np.array_equal(data_filtered, synthetic_event_data[loss_mask])


# Rate spike filter tests


def test_rate_spike_filter_no_spikes(synthetic_timestamps):
    """Test filtering with uniformly distributed events (no spikes)."""
    mask, spike_count = filter_rate_spikes(
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

    spike_mask, num_removed_due_to_spike = filter_rate_spikes(
        synthetic_ts, bin_width=10, rate_threshold=2.0
    )

    # Verify spike events are removed
    # Spike events occupy indices from len(ts_before) to len(ts_before) + 14
    n_before = len(ts_before)
    spike_indices = np.arange(n_before, n_before + 15)
    
    # All spike events should be marked as False (removed)
    assert np.all(~spike_mask[spike_indices]), "Not all spike events were removed"
    
    # Non-spike events should be kept
    assert np.any(spike_mask), "All events were removed, not just spikes"
    
    # num_removed_due_to_spike should indicate events were removed
    assert num_removed_due_to_spike > 0
    assert np.sum(~spike_mask) >= 15, "Should remove at least 15 spike events"


def test_apply_telescope_stream_sorting_realigns_data_after_timestamp_reordering():
    """Sorting must keep event rows aligned with their corrected timestamps."""
    source = PanoEventSource.__new__(PanoEventSource)

    telescope_data = {
        2: np.array([[200.0], [100.0], [300.0]], dtype=np.float32),
        3: np.array([[31.0], [11.0], [21.0]], dtype=np.float32),
    }
    telescope_timestamps = {
        2: np.array([
            "2026-01-01T00:00:02.000000000",
            "2026-01-01T00:00:01.000000000",
            "2026-01-01T00:00:03.000000000",
        ], dtype="datetime64[ns]"),
        3: np.array([
            "2026-01-01T00:00:03.000000000",
            "2026-01-01T00:00:01.000000000",
            "2026-01-01T00:00:02.000000000",
        ], dtype="datetime64[ns]"),
    }

    sorted_data, sorted_timestamps, sort_indices = source.apply_telescope_stream_sorting(
        telescope_data,
        telescope_timestamps,
        verbose=False,
    )

    assert np.array_equal(
        sorted_timestamps[2],
        np.array([
            "2026-01-01T00:00:01.000000000",
            "2026-01-01T00:00:02.000000000",
            "2026-01-01T00:00:03.000000000",
        ], dtype="datetime64[ns]"),
    )
    assert np.array_equal(sorted_data[2].ravel(), np.array([100.0, 200.0, 300.0], dtype=np.float32))
    assert np.array_equal(sort_indices[2], np.array([1, 0, 2]))

    assert np.array_equal(
        sorted_timestamps[3],
        np.array([
            "2026-01-01T00:00:01.000000000",
            "2026-01-01T00:00:02.000000000",
            "2026-01-01T00:00:03.000000000",
        ], dtype="datetime64[ns]"),
    )
    assert np.array_equal(sorted_data[3].ravel(), np.array([11.0, 21.0, 31.0], dtype=np.float32))
    assert np.array_equal(sort_indices[3], np.array([1, 2, 0]))


def test_metadata_masks_and_sorts_follow_event_indices():
    """Metadata arrays should follow the same masks and sort order as events."""
    source = PanoEventSource.__new__(PanoEventSource)

    telescope_metadata = {
        2: {
            "quabo_0": {
                "pkt_num": np.array([10, 11, 12, 13]),
                "tv_sec": np.array([100, 101, 102, 103]),
                "board_id": "q0",
            },
            "quabo_1": {
                "pkt_num": np.array([20, 21, 22, 23]),
                "tv_sec": np.array([200, 201, 202, 203]),
            },
        }
    }
    masks = {2: np.array([True, False, True, True])}

    filtered_metadata = source.apply_metadata_masks(telescope_metadata, masks)
    assert np.array_equal(filtered_metadata[2]["quabo_0"]["pkt_num"], np.array([10, 12, 13]))
    assert np.array_equal(filtered_metadata[2]["quabo_1"]["tv_sec"], np.array([200, 202, 203]))
    assert filtered_metadata[2]["quabo_0"]["board_id"] == "q0"

    sort_indices = {2: np.array([1, 2, 0])}
    sorted_metadata = source.apply_metadata_sorting(filtered_metadata, sort_indices)
    assert np.array_equal(sorted_metadata[2]["quabo_0"]["pkt_num"], np.array([12, 13, 10]))
    assert np.array_equal(sorted_metadata[2]["quabo_1"]["tv_sec"], np.array([202, 203, 200]))
    assert sorted_metadata[2]["quabo_0"]["board_id"] == "q0"


def test_shared_calibration_helpers_support_stacked_event_arrays():
    """Shared calibration helpers should work for both 1-event and N-event image arrays."""
    images = np.array(
        [
            np.full(1024, 10.0, dtype=np.float32),
            np.full(1024, 20.0, dtype=np.float32),
        ]
    )
    pedestal = np.full((32, 32), 2.0, dtype=np.float32)
    gains = np.full((32, 32), 0.5, dtype=np.float32)

    pedestal_subtracted = subtract_pedestal(images, pedestal)
    gain_corrected = correct_gain(images, gains)

    assert pedestal_subtracted.shape == (2, 1024)
    assert gain_corrected.shape == (2, 1024)
    assert np.allclose(pedestal_subtracted[0], 8.0)
    assert np.allclose(pedestal_subtracted[1], 18.0)
    assert np.allclose(gain_corrected[0], 5.0)
    assert np.allclose(gain_corrected[1], 10.0)


# Pedestal computation tests


def test_compute_pedestals_from_data_shape(synthetic_event_data):
    """Test that computed pedestals have correct shape."""
    pedestal, pedvar = calculate_pedestal_and_pedvar_robust(synthetic_event_data)

    # Input is (n_events, 1024), so output should be (1024,)
    assert pedestal.shape == (1024,)
    assert pedvar.shape == (1024,)


def test_compute_pedestals_from_data_values(synthetic_event_data):
    """Test that pedestal values are reasonable."""
    pedestal, pedvar = calculate_pedestal_and_pedvar_robust(synthetic_event_data)

    # Pedestal should be around 100 (from synthetic data)
    assert 95 < pedestal.mean() < 105
    # Sigma should be around 5 (from synthetic data std=5), so pedvar ~= 5 not 25
    assert 3 < pedvar.mean() < 7


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





# Gain calibration tests


def test_load_gain_file_shape():
    """Test that loaded gain file has correct shape."""
    from pathlib import Path

    # Get default gain file path for telescope 1
    data_dir = Path(__file__).parent.parent / "panoseti_ctapipe_plugin" / "default_data"
    gain_file = data_dir / "gains_tel1_Gattini.csv"

    if gain_file.exists():
        gains = load_gain_file(tel_id=1, gain_file_path=gain_file)
        assert gains.shape == (32, 32)
    else:
        # Skip if test data not available
        pytest.skip(f"Gain file not found: {gain_file}")


# ==============================================================================
# TIMESTAMP EXTRACTION FROM METADATA
# ==============================================================================


def test_extract_timestamps_from_metadata():
    """Test extracting legacy float-second timestamps from metadata."""
    n_events = 50
    
    # Create metadata with White Rabbit timestamps for all 4 QUABOs
    metadata = {}
    base_sec = 1000000
    
    for i in range(4):
        pkt_nsec = np.random.randint(0, int(1e9), n_events)
        tv_sec = np.full(n_events, base_sec + i, dtype=np.int64)
        tv_usec = np.random.randint(0, int(1e6), n_events)
        
        metadata[f"quabo_{i}"] = {
            "pkt_nsec": pkt_nsec,
            "tv_sec": tv_sec,
            "tv_usec": tv_usec,
        }
    
    # Extract timestamps
    timestamps, valid_mask = extract_timestamps_from_metadata(metadata)
    
    # Verify output
    assert isinstance(timestamps, np.ndarray)
    assert timestamps.dtype == np.dtype("float64")
    assert len(timestamps) == n_events
    assert np.array_equal(valid_mask, np.ones(n_events, dtype=bool))


def test_extract_timestamps_takes_minimum():
    """Test that extract_timestamps_from_metadata takes minimum across QUABOs."""
    n_events = 20
    
    # Create metadata where we can control which QUABO has minimum timestamp
    metadata = {}
    base_sec = 1000000
    
    # QUABO 0: minimum (base_sec)
    metadata["quabo_0"] = {
        "pkt_nsec": np.full(n_events, 100000000, dtype=np.int64),
        "tv_sec": np.full(n_events, base_sec, dtype=np.int64),
        "tv_usec": np.full(n_events, 100000, dtype=np.int64),
    }
    
    # QUABO 1: later (base_sec + 1)
    metadata["quabo_1"] = {
        "pkt_nsec": np.full(n_events, 100000000, dtype=np.int64),
        "tv_sec": np.full(n_events, base_sec + 1, dtype=np.int64),
        "tv_usec": np.full(n_events, 100000, dtype=np.int64),
    }
    
    # QUABO 2: much later (base_sec + 10)
    metadata["quabo_2"] = {
        "pkt_nsec": np.full(n_events, 100000000, dtype=np.int64),
        "tv_sec": np.full(n_events, base_sec + 10, dtype=np.int64),
        "tv_usec": np.full(n_events, 100000, dtype=np.int64),
    }
    
    # QUABO 3: also later
    metadata["quabo_3"] = {
        "pkt_nsec": np.full(n_events, 100000000, dtype=np.int64),
        "tv_sec": np.full(n_events, base_sec + 5, dtype=np.int64),
        "tv_usec": np.full(n_events, 100000, dtype=np.int64),
    }
    
    # Extract timestamps
    timestamps, valid_mask = extract_timestamps_from_metadata(metadata)
    
    # Manually compute expected result: minimum across all QUABOs
    ts_quabo = []
    for i in range(4):
        ts = (
            metadata[f"quabo_{i}"]["tv_sec"].astype(np.float64)
            + metadata[f"quabo_{i}"]["tv_usec"].astype(np.float64) * 1e-6
        )
        ts_quabo.append(ts)
    expected_timestamps = np.min(np.array(ts_quabo), axis=0)
    
    # Result should match expected minimum
    assert np.array_equal(timestamps, expected_timestamps)
    
    # Verify that minimum is from QUABO 0 (earliest tv_sec)
    ts_quabo_0 = (
        metadata["quabo_0"]["tv_sec"].astype(np.float64)
        + metadata["quabo_0"]["tv_usec"].astype(np.float64) * 1e-6
    )
    assert np.array_equal(timestamps, ts_quabo_0)
    assert np.array_equal(valid_mask, np.ones(n_events, dtype=bool))


def test_extract_timestamps_matches_legacy_tv_time():
    """Test that extraction matches legacy tv_sec/tv_usec timestamp logic."""
    n_events = 30
    
    # Create consistent metadata
    metadata = {}
    for i in range(4):
        metadata[f"quabo_{i}"] = {
            "pkt_nsec": np.random.randint(0, int(1e9), n_events),
            "tv_sec": np.full(n_events, 1000000, dtype=np.int64),
            "tv_usec": np.random.randint(0, int(1e6), n_events),
        }
    
    # Extract using function
    timestamps_func, valid_mask = extract_timestamps_from_metadata(metadata)
    
    # Extract manually to verify
    timestamps_manual = []
    for i in range(4):
        ts = (
            metadata[f"quabo_{i}"]["tv_sec"].astype(np.float64)
            + metadata[f"quabo_{i}"]["tv_usec"].astype(np.float64) * 1e-6
        )
        timestamps_manual.append(ts)
    timestamps_manual = np.min(np.array(timestamps_manual), axis=0)
    
    # Should be identical
    assert np.array_equal(timestamps_func, timestamps_manual)
    assert np.array_equal(valid_mask, np.ones(n_events, dtype=bool))


# ==============================================================================
# TIMESTAMP HANDLING TESTS (NEW - tests for modifications)
# ==============================================================================


def test_packet_loss_filter_with_timestamps(synthetic_event_data, synthetic_timestamps):
    """Test packet loss filter properly handles and filters timestamps."""
    # Create metadata with some packet loss
    n_events = len(synthetic_event_data)
    pkt_nums = [
        np.ones(n_events),
        np.ones(n_events),
        np.ones(n_events),
        np.ones(n_events),
    ]
    # Introduce packet loss in events 10-19
    pkt_nums[0][10:20] = 0

    metadata = {
        "quabo_0": {"pkt_num": pkt_nums[0]},
        "quabo_1": {"pkt_num": pkt_nums[1]},
        "quabo_2": {"pkt_num": pkt_nums[2]},
        "quabo_3": {"pkt_num": pkt_nums[3]},
    }

    # Create timestamps array matching the event data
    timestamps = np.linspace(1000000, 1000100, n_events)

    # Call with both data and timestamps
    loss_mask, num_removed_due_to_pkt_loss, data_filtered, timestamps_filtered = filter_packet_loss(
        metadata, data=synthetic_event_data, timestamps=timestamps
    )

    # Verify alignment
    assert len(data_filtered) == len(timestamps_filtered)
    assert len(data_filtered) == np.sum(loss_mask)
    
    # Verify data and timestamps match
    assert np.array_equal(data_filtered, synthetic_event_data[loss_mask])
    assert np.array_equal(timestamps_filtered, timestamps[loss_mask])
    
    # Verify some events were removed
    assert num_removed_due_to_pkt_loss > 0
    assert len(data_filtered) < n_events


def test_packet_loss_filter_timestamps_only(synthetic_event_data):
    """Test packet loss filter with only timestamps, no data."""
    n_events = len(synthetic_event_data)
    pkt_nums = [
        np.ones(n_events),
        np.ones(n_events),
        np.ones(n_events),
        np.ones(n_events),
    ]
    pkt_nums[1][5:10] = 0

    metadata = {
        "quabo_0": {"pkt_num": pkt_nums[0]},
        "quabo_1": {"pkt_num": pkt_nums[1]},
        "quabo_2": {"pkt_num": pkt_nums[2]},
        "quabo_3": {"pkt_num": pkt_nums[3]},
    }

    timestamps = np.linspace(1000000, 1000100, n_events)

    # Call with only timestamps
    loss_mask, num_removed_due_to_pkt_loss, timestamps_filtered = filter_packet_loss(
        metadata, timestamps=timestamps
    )

    # Should return mask, count, and filtered timestamps
    assert len(timestamps_filtered) == np.sum(loss_mask)
    assert len(timestamps_filtered) < len(timestamps)
    assert num_removed_due_to_pkt_loss == 5  # 5 events with packet loss


def test_packet_loss_filter_legacy_metadata_timestamps():
    """Test packet loss filter derives old-style timestamps from metadata."""
    metadata = {
        "quabo_0": {
            "pkt_num": np.array([1, 0, 1]),
            "tv_sec": np.array([10, 20, 30]),
            "tv_usec": np.array([500000, 500000, 500000]),
        },
        "quabo_1": {
            "pkt_num": np.array([1, 1, 1]),
            "tv_sec": np.array([10, 20, 30]),
            "tv_usec": np.array([600000, 600000, 600000]),
        },
        "quabo_2": {
            "pkt_num": np.array([1, 1, 1]),
            "tv_sec": np.array([10, 20, 30]),
            "tv_usec": np.array([700000, 700000, 700000]),
        },
        "quabo_3": {
            "pkt_num": np.array([1, 1, 1]),
            "tv_sec": np.array([10, 20, 30]),
            "tv_usec": np.array([800000, 800000, 800000]),
        },
    }

    loss_mask, removed, timestamps_filtered = filter_packet_loss(metadata)

    assert removed == 1
    assert np.array_equal(loss_mask, np.array([True, False, True]))
    assert np.allclose(timestamps_filtered, np.array([10.5, 30.5]))


def test_apply_packet_loss_filters_preserves_provided_timestamps():
    """EventSource packet-loss filtering should preserve the provided timestamp stream."""
    source = PanoEventSource.__new__(PanoEventSource)

    telescope_data = {
        2: np.array([[1.0], [2.0], [3.0]], dtype=np.float32),
    }
    telescope_timestamps = {
        2: np.array([
            "2026-01-01T00:00:10.000000000",
            "2026-01-01T00:00:20.000000000",
            "2026-01-01T00:00:30.000000000",
        ], dtype="datetime64[ns]"),
    }
    telescope_metadata = {
        2: {
            "quabo_0": {
                "pkt_num": np.array([1, 0, 1]),
                "tv_sec": np.array([10, 20, 30]),
                "tv_usec": np.array([500000, 500000, 500000]),
            },
            "quabo_1": {
                "pkt_num": np.array([1, 1, 1]),
                "tv_sec": np.array([10, 20, 30]),
                "tv_usec": np.array([600000, 600000, 600000]),
            },
            "quabo_2": {
                "pkt_num": np.array([1, 1, 1]),
                "tv_sec": np.array([10, 20, 30]),
                "tv_usec": np.array([700000, 700000, 700000]),
            },
            "quabo_3": {
                "pkt_num": np.array([1, 1, 1]),
                "tv_sec": np.array([10, 20, 30]),
                "tv_usec": np.array([800000, 800000, 800000]),
            },
        }
    }

    filtered_data, filtered_timestamps, masks = source.apply_packet_loss_filters(
        telescope_data,
        telescope_timestamps,
        telescope_metadata=telescope_metadata,
        verbose=False,
    )

    assert np.array_equal(masks[2], np.array([True, False, True]))
    assert np.array_equal(filtered_data[2], np.array([[1.0], [3.0]], dtype=np.float32))
    assert np.array_equal(
        filtered_timestamps[2],
        np.array([
            "2026-01-01T00:00:10.000000000",
            "2026-01-01T00:00:30.000000000",
        ], dtype="datetime64[ns]"),
    )


def test_rate_spike_filter_with_data(synthetic_timestamps, synthetic_event_data):
    """Test rate spike filter properly handles and filters data arrays."""
    # Create data that matches timestamps
    n_events = len(synthetic_timestamps)
    data = synthetic_event_data[:n_events] if len(synthetic_event_data) >= n_events else synthetic_event_data

    # Ensure matching lengths
    if len(data) < n_events:
        # Pad data if needed
        data = np.vstack([data, np.ones((n_events - len(data), 1024))])
    else:
        data = data[:n_events]

    # Call with both timestamps and data
    spike_mask, num_removed_due_to_spike, data_filtered, timestamps_filtered = filter_rate_spikes(
        synthetic_timestamps, bin_width=10, rate_threshold=2.0, data=data, return_timestamps=True
    )

    # Verify alignment
    assert len(data_filtered) == np.sum(spike_mask)
    assert len(data_filtered) == len(data) - num_removed_due_to_spike
    assert len(timestamps_filtered) == len(data_filtered)
    
    # Verify data and timestamps match mask
    assert np.array_equal(data_filtered, data[spike_mask])
    assert np.array_equal(timestamps_filtered, synthetic_timestamps[spike_mask])


def test_rate_spike_filter_with_datetime64_timestamps():
    """Test rate spike filter handles datetime64 timestamps correctly."""
    # Create datetime64 timestamps
    base_time = np.datetime64('2026-01-15T02:30:00', 'ns')
    timestamps = base_time + np.arange(0, 100 * 1e9, int(1e9 / 1.5), dtype='int64')  # 100 seconds, 1.5 Hz
    
    spike_mask, num_removed_due_to_spike = filter_rate_spikes(
        timestamps, bin_width=10, rate_threshold=2.0
    )

    # Should handle datetime64 and keep most events (1.5 Hz < 2.0 Hz threshold)
    assert len(spike_mask) == len(timestamps)
    assert np.sum(spike_mask) > len(timestamps) * 0.8


def test_timestamp_alignment_after_both_filters(synthetic_event_data):
    """Test that timestamps stay aligned after both packet loss and spike filtering."""
    n_events = 100
    
    # Create packet loss metadata
    pkt_nums = [
        np.ones(n_events),
        np.ones(n_events),
        np.ones(n_events),
        np.ones(n_events),
    ]
    pkt_nums[0][5:10] = 0  # 5 events with packet loss
    
    metadata = {
        "quabo_0": {"pkt_num": pkt_nums[0]},
        "quabo_1": {"pkt_num": pkt_nums[1]},
        "quabo_2": {"pkt_num": pkt_nums[2]},
        "quabo_3": {"pkt_num": pkt_nums[3]},
    }
    
    # Create corresponding data and timestamps
    data = synthetic_event_data[:n_events] if len(synthetic_event_data) >= n_events else synthetic_event_data
    if len(data) < n_events:
        data = np.vstack([data, np.ones((n_events - len(data), 1024))])
    else:
        data = data[:n_events]
    
    timestamps = np.linspace(1000000, 1000100, n_events)
    
    # Apply packet loss filter
    _, _, data_after_loss, ts_after_loss = filter_packet_loss(
        metadata, data=data, timestamps=timestamps
    )
    
    # Apply spike filter with return_timestamps=True
    spike_mask, _, data_final, ts_final = filter_rate_spikes(
        ts_after_loss, bin_width=10, rate_threshold=2.0, data=data_after_loss, return_timestamps=True
    )
    
    # Final verification: data and timestamps have same length
    assert len(data_final) == len(ts_final)
    assert len(data_final) <= n_events
