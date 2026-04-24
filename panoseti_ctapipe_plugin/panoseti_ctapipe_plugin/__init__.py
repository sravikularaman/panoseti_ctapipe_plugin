"""
PANOSETI ctapipe plugin.

This package provides integration between PANOSETI PFF pulse height data
and the ctapipe gamma-ray analysis framework.

Main components:
- PanoEventSource: EventSource for reading PANOSETI PFF files

- Utility functions: Timestamp conversion, filtering, calibration
- Instrument description: Camera geometry, telescope layout

Author: Sruthi Ravikularaman
Last modified: 17 April 2026
"""

import logging

# Import main classes
from .eventsource import CalibrationPipeline, PanoEventSource

# Import instrument definitions
from .instrument import (
    camera,
    optics,
    subarray,
    telescope_1,
    telescope_2,
    telescope_3,
    telescope_4,
    MODULE_TO_TEL_ID,
)

# Import utility functions
from .functions import (
    apply_gain_correction,
    apply_rate_spike_filter,
    calculate_pedestal_and_pedvar_robust,
    calibrate_image,
    load_gain_file,
    subtract_pedestal,
    wr_to_unix,
)

__all__ = [
    # Main classes
    "CalibrationPipeline",
    "PanoEventSource",
    # Instrument
    "camera",
    "optics",
    "subarray",
    "telescope_1",
    "telescope_2",
    "telescope_3",
    "telescope_4",
    "MODULE_TO_TEL_ID",
    # Calibration functions
    "apply_gain_correction",
    "calibrate_image",
    "subtract_pedestal",
    # Rate spike filtering
    "apply_rate_spike_filter",
    # Utility functions
    "calculate_pedestal_and_pedvar_robust",
    "load_gain_file",
    "wr_to_unix",
]

# Configure logging
logger = logging.getLogger(__name__)
logger.addHandler(logging.NullHandler())

__version__ = "0.1.0"
