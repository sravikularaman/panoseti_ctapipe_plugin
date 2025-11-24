"""
PANOSETI ctapipe plugin
Provides:
- PanoEventSource for reading .pff data
- PanoReconstructor 
- Instrument description (optics, camera, subarray)
"""

from .instrument import subarray
from .eventsource import PanoEventSource
from .reconstructor import PanoReconstructor

__all__ = ["subarray", "PanoEventSource", "PanoReconstructor"]
__version__ = "0.1.0"