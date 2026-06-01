"""
PANOSETI ctapipe plugin.

This package provides integration between PANOSETI PFF pulse height data
and the ctapipe gamma-ray analysis framework.

Main components:
- PanoEventSource: EventSource for reading PANOSETI PFF files
- Utility functions: Timestamp conversion, filtering, calibration
- Instrument description: Camera geometry, telescope layout

Last modified: 6 May 2026
"""

import sys

# Make submodules accessible
from . import plugin_src
from .plugin_src import instrument, eventsource, functions

# Register submodules in sys.modules so "from panoseti_ctapipe_plugin.instrument import ..." works
sys.modules['panoseti_ctapipe_plugin.instrument'] = instrument
sys.modules['panoseti_ctapipe_plugin.eventsource'] = eventsource
sys.modules['panoseti_ctapipe_plugin.functions'] = functions

# Import from plugin_src
from .plugin_src import (
    PanoEventSource,
    camera,
    correct_gain,
    correct_telescope_timing,
    filter_packet_loss,
    filter_rate_spikes,
    geometry,
    optics,
    subarray,
    telescope_1,
    telescope_2,
    telescope_3,
    telescope_4,
    MODULE_TO_TEL_ID,
    calculate_pedestal_and_pedvar_robust,
    calibrate_image,
    load_gain_file,
    subtract_pedestal,
    wr_to_unix,
)

__all__ = [
    # Submodules
    "instrument",
    "eventsource",
    "functions",
    # Main classes
    "PanoEventSource",
    # Instrument
    "camera",
    "geometry",
    "optics",
    "subarray",
    "telescope_1",
    "telescope_2",
    "telescope_3",
    "telescope_4",
    "MODULE_TO_TEL_ID",
    # Utility functions
    "correct_gain",
    "correct_telescope_timing",
    "filter_packet_loss",
    "filter_rate_spikes",
    "calculate_pedestal_and_pedvar_robust",
    "calibrate_image",
    "load_gain_file",
    "subtract_pedestal",
    "wr_to_unix",
]

__version__ = "0.1.0"
