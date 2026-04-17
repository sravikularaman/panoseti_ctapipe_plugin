"""
Shared pytest fixtures and configuration for all PANOSETI plugin tests.

This file is automatically discovered by pytest and provides shared fixtures
and configuration for all test modules.

Following ctapipe guidelines:
https://ctapipe.readthedocs.io/en/stable/developer-guide/code-guidelines.html#unit-tests

Author: Sruthi Ravikularaman
Last modified: 17 April 2026
"""

from pathlib import Path

import numpy as np
import pytest


# ==============================================================================
# Path Fixtures
# ==============================================================================


@pytest.fixture
def test_data_dir():
    """Fixture providing path to test data directory."""
    return Path(__file__).parent.parent / "test_data"


@pytest.fixture
def test_obs_folder(test_data_dir):
    """Fixture providing path to a sample PFF observation folder."""
    obs_folder = (
        test_data_dir
        / "pff"
        / "obs_Palomar.start_2026-01-15T02:26:39Z.runtype_obs-test.pffd"
        / "obs_Palomar.start_2026-01-15T02:26:39Z.runtype_obs-test.pffd"
    )
    if not obs_folder.exists():
        pytest.skip(f"Test data not found: {obs_folder}")
    return obs_folder


# ==============================================================================
# Data Fixtures
# ==============================================================================


@pytest.fixture
def synthetic_timestamps():
    """Fixture providing synthetic event timestamps."""
    # 150 events over 100 seconds
    return np.linspace(1000000, 1000100, 150)


@pytest.fixture
def synthetic_event_data():
    """Fixture providing synthetic 32x32 camera image data."""
    np.random.seed(42)
    n_events = 100
    # Generate synthetic data: (n_events, 1024) flattened images
    mean_pedestal = 100
    pedvar = 5
    data = mean_pedestal + np.random.normal(0, pedvar, (n_events, 1024))
    return data.astype(np.float32)


@pytest.fixture
def synthetic_waveform_data():
    """Fixture providing synthetic raw waveform-like data."""
    np.random.seed(42)
    n_events = 50
    # Simulate raw ADC counts
    data = 50 + np.random.randint(0, 100, (n_events, 1024))
    return data.astype(np.uint16)


# ==============================================================================
# Instrument Fixtures
# ==============================================================================


@pytest.fixture
def default_subarray():
    """Fixture providing the default PANOSETI subarray description."""
    from src import subarray

    return subarray


@pytest.fixture
def default_camera():
    """Fixture providing the default PANOSETI camera description."""
    from src import camera

    return camera


@pytest.fixture
def default_optics():
    """Fixture providing the default PANOSETI optics description."""
    from src import optics

    return optics


# ==============================================================================
# Marker Configuration
# ==============================================================================


def pytest_configure(config):
    """Register custom pytest markers."""
    config.addinivalue_line(
        "markers",
        "requires_test_data: mark test as requiring real PFF test data",
    )
    config.addinivalue_line(
        "markers",
        "slow: mark test as slow running",
    )
    config.addinivalue_line(
        "markers",
        "integration: mark test as an integration test",
    )
