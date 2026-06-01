"""
Utility functions for PANOSETI data processing.

This module re-exports functions organized into topic-specific submodules in the sibling utils package:
- Calibration: gain calibration and image calibration
- Timing: timestamp conversion and telescope synchronization
- Data filtering: packet loss and rate spike filtering
- Coincidence: multi-telescope event matching
- Pedestal/Pedvar: pedestal computation with outlier removal
- Pointing offset: pixel-to-sky coordinate conversion
- Meridian flip: image rotation correction

For direct imports of specific submodules, use:
    from utils.calibration import load_gain_file
    from utils.timing import wr_to_unix
    etc.

Or via the re-export hub:
    from plugin_src.functions import load_gain_file, wr_to_unix

Last modified: 22 April 2026
"""

from ..utils.calibration import (
    load_gain_file,
    subtract_pedestal,
    correct_gain,
    calibrate_image,
)
from ..utils.timing import (
    wr_to_unix,
    extract_timestamps_from_metadata,
    measure_telescope_timing_offset,
    correct_telescope_timing,
)
from ..utils.data_filtering import (
    filter_packet_loss,
    filter_rate_spikes,
)
from ..utils.coincidence import (
    find_two_telescope_coincident_events,
    find_multi_telescope_coincident_events,
    calculate_coincidence_rate,
    match_coincident_events,
)
from ..utils.pedestal_pedvar import (
    _gaussian,
    calculate_pedestal_and_pedvar_robust,
)
from ..utils.pointing_offset import (
    load_pointing_offset_csv,
    pixel_to_skycoord,
    get_pointing_offset_for_observation,
)
from ..utils.meridian_flip import (
    rotate_images_after_meridian_flip,
)

__all__ = [
    # Calibration
    "load_gain_file",
    "subtract_pedestal",
    "correct_gain",
    "calibrate_image",
    # Timing
    "wr_to_unix",
    "extract_timestamps_from_metadata",
    "measure_telescope_timing_offset",
    "correct_telescope_timing",
    # Data filtering
    "filter_packet_loss",
    "filter_rate_spikes",
    # Coincidence
    "find_two_telescope_coincident_events",
    "find_multi_telescope_coincident_events",
    "calculate_coincidence_rate",
    "match_coincident_events",
    # Pedestal/Pedvar
    "_gaussian",
    "calculate_pedestal_and_pedvar_robust",
    # Pointing offset
    "load_pointing_offset_csv",
    "pixel_to_skycoord",
    "get_pointing_offset_for_observation",
    # Meridian flip
    "rotate_images_after_meridian_flip",
]
