"""
Instrument description for PANOSETI telescopes.

Defines camera geometry, optics, telescope descriptions, and subarray layout.

Author: Sruthi Ravikularaman
Last modified: 17 April 2026
"""

import astropy.units as u
from astropy.coordinates import EarthLocation

from ctapipe.instrument import (
    CameraDescription,
    CameraGeometry,
    CameraReadout,
    OpticsDescription,
    ReflectorShape,
    SizeType,
    SubarrayDescription,
    TelescopeDescription,
)
from ctapipe.coordinates import CameraFrame
import numpy as np

__all__ = [
    "camera",
    "optics",
    "telescope_1",
    "telescope_2",
    "telescope_3",
    "telescope_4",
    "subarray",
    "MODULE_TO_TEL_ID",
]

# ==============================================================================
# CAMERA AND OPTICS DESCRIPTION
# ==============================================================================

# Camera geometry parameters
nx, ny = 32, 32
pix_size = 10.0 / 32 * u.deg

# Focal length of the lens
focal_length = 0.46 * u.m

# Actual Hamamatsu detector measurements (from S13361-3050AE-08 datasheet)
# Each pixel: 3.0 mm × 3.0 mm
# 8×8 pixels per detector module, 4×4 modules per telescope = 32×32 total pixels
pixel_pitch_mm = 3.0  # mm
pixel_pitch_m = pixel_pitch_mm / 1000.0  # convert to meters

# Create rectangular pixel layout for 32x32 grid
# Centered at origin
x = np.arange(32) * pixel_pitch_m - (32 - 1) * pixel_pitch_m / 2
y = np.arange(32) * pixel_pitch_m - (32 - 1) * pixel_pitch_m / 2
xx, yy = np.meshgrid(x, y)
pix_x = xx.flatten()
pix_y = yy.flatten()

pixel_size_m = pixel_pitch_m

# Create camera geometry with explicit pixel positions
geometry = CameraGeometry(
    name="PANOSETI",
    pix_id=np.arange(1024),
    pix_x=pix_x * u.m,
    pix_y=pix_y * u.m,
    pix_area=np.ones(1024) * (pixel_pitch_m ** 2) * u.m**2,
    pix_type="square",  # Square pixels for 32x32 grid layout
    neighbors=None,  # Will be computed automatically
)

# Camera readout parameters
readout = CameraReadout(
    name="Panoseti",
    n_pixels=1024,
    n_channels=1,
    reference_pulse_shape=np.array([[]]),
    reference_pulse_sample_width=1.0 * u.ns,
    sampling_rate=1.0 * u.GHz,
    n_samples=1,
    n_samples_long=1,
)

# Camera description
camera = CameraDescription(
    name="Panoseti",
    geometry=geometry,
    readout=readout,
)

# Optics parameters
aperture = 0.46 * u.m
radius = aperture / 2.0
lens_area = 3.1416 * radius * radius

optics = OpticsDescription(
    name="Panoseti_Fresnel",
    size_type=SizeType.UNKNOWN,
    n_mirrors=1,
    n_mirror_tiles=1,
    equivalent_focal_length=focal_length,
    effective_focal_length=focal_length,
    mirror_area=lens_area,
    # PANOSETI uses Fresnel lenses, not reflective mirrors.
    # Once ctapipe supports ReflectorShape.FRESNEL, change from UNKNOWN
    reflector_shape=ReflectorShape.UNKNOWN,  # TODO: FRESNEL when available in ctapipe
)

# Set camera frame with focal length
geometry.frame = CameraFrame(focal_length=optics.effective_focal_length)

# ==============================================================================
# TELESCOPE AND SUBARRAY DESCRIPTION
# ==============================================================================

# Telescope descriptions (all use same camera and optics for now)
telescope_1 = TelescopeDescription(name="Gattini", optics=optics, camera=camera)
telescope_2 = TelescopeDescription(name="Winter", optics=optics, camera=camera)
telescope_3 = TelescopeDescription(name="Fern", optics=optics, camera=camera)
telescope_4 = TelescopeDescription(name="PTI-Heli", optics=optics, camera=camera)

# Subarray description with telescope positions at Palomar Observatory
subarray = SubarrayDescription(
    name="Panoseti-Palomar",
    tel_descriptions={
        1: telescope_1,
        2: telescope_2,
        3: telescope_3,
        4: telescope_4,
    },
    tel_positions={
        1: [177.58, -333.33, 42.82] * u.m,   # Gattini (CORSIKA coordinates)
        2: [-220.15, 33.67, 49.17] * u.m,    # Winter (CORSIKA coordinates)
        3: [-130.43, 190.40, 34.66] * u.m,   # Fern (CORSIKA coordinates)
        4: [-1.0, 97.18, 39.70] * u.m,       # PTI-Heli (CORSIKA coordinates)
    },
    reference_location=EarthLocation(
        lat=33.3564 * u.deg,
        lon=-116.865 * u.deg,
        height=1712 * u.m,
    ),
)

# Mapping of PFF module numbers to telescope IDs
MODULE_TO_TEL_ID = {
    254: 1,  # Gattini
    253: 2,  # Winter
    252: 3,  # Fern
    250: 4,  # PTI-Heli
}
