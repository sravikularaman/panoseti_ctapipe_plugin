from astropy import units as u
from astropy.coordinates import EarthLocation
from ctapipe.instrument import (
    CameraGeometry, CameraDescription, OpticsDescription,
    ReflectorShape, TelescopeDescription, SubarrayDescription
)

# Camera: 64x64, 10° FoV
nx, ny = 64, 64
pix_size = 10.0 / 64 * u.deg
geometry = CameraGeometry.make_rectangular(nx=nx, ny=ny, pix_x=pix_size, pix_y=pix_size)
camera = CameraDescription(name="Panoseti", geometry=geometry)

# Optics: Fresnel lens f/1, 0.5 m aperture
aperture = 0.5 * u.m
focal_length = aperture
lens_area = 3.1416 * (aperture / 2)**2

optics = OpticsDescription(
    name="Panoseti",
    size_type=None,
    n_mirrors=1,
    equivalent_focal_length=focal_length,
    mirror_area=lens_area,
    reflector_shape=ReflectorShape.PARABOLIC
)

telescope = TelescopeDescription(name="Panoseti", optics=optics, camera=camera)

subarray = SubarrayDescription(
    name="Panoseti",
    tel_descriptions={1: telescope},
    tel_positions={1: [0, 0, 0]*u.m},
    reference_location=EarthLocation(lat=0*u.deg, lon=0*u.deg, height=100*u.m)
)
