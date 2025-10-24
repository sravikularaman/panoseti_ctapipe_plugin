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

# Optics: Fresnel lens f/1, 0.46 m aperture
aperture = 0.46 * u.m
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

# Telescopes
telescope_1 = TelescopeDescription(name="Gattini" type="PANOSETI", optics=optics, camera=camera)

telescope_2 = TelescopeDescription(name="Winter", type="PANOSETI", optics=optics, camera=camera)

subarray = SubarrayDescription(
    name="Panoseti-Palomar",
    tel_descriptions={1: telescope_1, 2: telescope_2},
    tel_positions={1: [0, 0, 0]*u.m, 2: [10, 0, 0]*u.m}, # To change with actual coordinates
    reference_location = EarthLocation(
    lat=33.3564 * u.deg,
    lon=-116.865 * u.deg,
    height=1712 * u.m
)
)
