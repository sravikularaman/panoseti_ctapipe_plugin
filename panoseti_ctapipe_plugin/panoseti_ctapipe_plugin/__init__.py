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
from .eventsource import PanoEventSource

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
    apply_packet_loss_filter,
    apply_rate_spike_filter,
    calculate_pedestal_and_pedvar_robust,
    compute_pedestals_from_data,
    load_gain_file,
    select_time_interval,
    wr_to_unix,
)

__all__ = [
    # Main classes
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
    # Functions
    "apply_packet_loss_filter",
    "apply_rate_spike_filter",
    "calculate_pedestal_and_pedvar_robust",
    "compute_pedestals_from_data",
    "load_gain_file",
    "select_time_interval",
    "wr_to_unix",
]

# Configure logging
logger = logging.getLogger(__name__)
logger.addHandler(logging.NullHandler())

__version__ = "0.1.0"
