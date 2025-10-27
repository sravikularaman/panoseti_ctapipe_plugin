"""
PANOSETI ctapipe plugin
Provides:
- PanoEventSource for reading .pff data
- Instrument description (optics, camera, subarray)
"""

from .instrument import subarray
from .eventsource import PanoEventSource

__all__ = ["subarray", "PanoEventSource"]
__version__ = "0.1.0"