from astropy import units as u
from astropy.coordinates import EarthLocation
from ctapipe.instrument import (
    CameraGeometry, CameraDescription, OpticsDescription,
    ReflectorShape, TelescopeDescription, SubarrayDescription
)

# Camera: 32x32, 10° FoV
nx, ny = 32, 32
pix_size = 10.0 / 32 * u.deg
geometry = CameraGeometry.make_rectangular(
    nx=32, ny=32,
    width=10.0 * u.deg,
    height=10.0 * u.deg
)
camera = CameraDescription(name="Panoseti", geometry=geometry)

# Optics: Fresnel lens f/1, 0.5 m aperture
aperture = 0.5 * u.m
focal_length = aperture
lens_area = 3.1416 * (aperture / 2)**2

optics = OpticsDescription(
    name="Panoseti_Fresnel",
    size_type=None,
    n_mirrors=1,
    equivalent_focal_length=focal_length,
    #effective_focal_length=
    mirror_area=lens_area,
    reflector_shape=ReflectorShape.UNKNOWN
)

telescope = TelescopeDescription(name="Panoseti", optics=optics, camera=camera)

subarray = SubarrayDescription(
    name="Panoseti",
    tel_descriptions={1: telescope},
    tel_positions={1: [0, 0, 0]*u.m},
    reference_location = EarthLocation(
    lat=33.3564 * u.deg,
    lon=-116.865 * u.deg,
    height=1712 * u.m
)
)
