import numpy as np
import pypff

import astropy.units as u
from astropy.coordinates import EarthLocation
from astropy.time import Time

from ctapipe.containers import (
    ObservationBlockContainer,
    ReconstructedGeometryContainer,
    SchedulingBlockContainer,
    TriggerContainer,
    ArrayEventContainer,
)
from ctapipe.core import traits
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
from ctapipe.io import DataLevel, EventSource
from ctapipe.reco import Reconstructor

from ctapipe.coordinates import CameraFrame

__all__ = [
    "PanoEventSource", 
    "PanoReconstructor",
    ]

# Camera: 32x32, 10° FoV
nx, ny = 32, 32
pix_size = 10.0 / 32 * u.deg


geometry = CameraGeometry.make_rectangular(
    npix_x=32, npix_y=32,
    range_x=(-0.04, 0.04), # Tuple of coordinates
    range_y=(-0.04, 0.04)
)

readout = CameraReadout(name='panoseti', n_pixels=1024, n_channels=1, reference_pulse_shape=np.array([[]]), reference_pulse_sample_width=1.0*u.ns, sampling_rate=1.0*u.GHz, n_samples=1, n_samples_long=1)

camera = CameraDescription(name="Panoseti", geometry=geometry, readout=readout)

# Optics: Fresnel lens f/1, 0.46 m aperture
aperture = 0.46 * u.m
focal_length = aperture
lens_area = 3.1416 * (aperture / 2)**2

optics = OpticsDescription(
    name="Panoseti_Fresnel",
    size_type=SizeType.SST,
    n_mirrors=1,
    n_mirror_tiles=1,
    equivalent_focal_length=focal_length,
    effective_focal_length=focal_length,
    mirror_area=lens_area,
    reflector_shape=ReflectorShape.UNKNOWN
)

geometry.frame = CameraFrame(focal_length=optics.effective_focal_length)
# Telescopes
telescope_1 = TelescopeDescription(name="Gattini", optics=optics, camera=camera)

telescope_2 = TelescopeDescription(name="Winter", optics=optics, camera=camera)

subarray = SubarrayDescription(
    name="Panoseti-Palomar",
    tel_descriptions={1: telescope_1, 2: telescope_2},
    tel_positions={1: [0, 0, 0]*u.m, 2: [1, 1, 0]*u.m}, # To change with actual coordinates
    reference_location = EarthLocation(
    lat=33.3564 * u.deg,
    lon=-116.865 * u.deg,
    height=1712 * u.m
)
)


class PanoEventSource(EventSource):
    is_simulation = False
    datalevels = (DataLevel.DL1_IMAGES,) # Images not waveforms
    subarray = subarray

    @classmethod
    def is_compatible(cls, path):
        return str(path).endswith(".pff")

    def _generator(self):

        # List all telescope files
        module_files = [
            str(self.input_url).replace("module_1", f"module_{i}") for i in range(1, 3)  # telescope 1 and 2 for now
            ]

        # Read all telescopes' data and metadata
        telescope_data = []
        telescope_metadata = []

        for file_path in module_files:
            pff_file = pypff.io.datapff(file_path)
            data, metadata = pff_file.readpff(metadata=True)
            telescope_data.append(data)
            telescope_metadata.append(metadata)

        # Collect timestamps and filter packet losses for each telescope
        all_timestamps_by_telescope = []
        valid_event_mask_by_telescope = [] 

        for metadata in telescope_metadata:
            quabo_timestamps = []
            quabo_valid = []

            for i in range(4):
                meta = metadata[f'quabo_{i}']
                timestamps = meta['tv_sec'] + meta['tv_usec'] * 1e-6

                # Filter out lost packets
                wout_pkt_loss = meta['pkt_num'] != 0
                timestamps = timestamps[wout_pkt_loss]

                quabo_timestamps.append(timestamps)
                quabo_valid.append(wout_pkt_loss)

            all_timestamps_by_telescope.append(quabo_timestamps)

            valid_events = np.all(np.column_stack(quabo_valid), axis=1)
            valid_event_mask_by_telescope.append(valid_events)
       
        # Loop through events

        num_events = min(len(data) for data in telescope_data) # Better way to loop?

        for i in range(num_events):

            if not all(valid_event_mask_by_telescope[tel][i]
                       for tel in range(len(valid_event_mask_by_telescope))):
                continue
            
            trigger = TriggerContainer(time=Time(meta['tv_sec'], meta['tv_usec']*1e-6, format='unix')) # Check format
            event = ArrayEventContainer(trigger=trigger)
            event.count = i
            event.trigger.tels_with_trigger = []


            for tel_id, data in enumerate(telescope_data, start=1):
                raw_event = data[i]

                event.r0.tel[tel_id].waveform = np.array(raw_event)[np.newaxis, :, np.newaxis]
                # ctapipe wants (n_channels, n_pixels, n_samples) and raw_event is (n_pixels) which is an element from data which is (n_samples, n_pixels)

                # Timestamp across quabos
                quabo_timestamps = all_timestamps_by_telescope[tel_id - 1]
                if all(len(ts) > i for ts in quabo_timestamps):
                    
                    event_time = np.min([ts[i] for ts in quabo_timestamps])
                    # event_time = np.median([ts[i] for ts in quabo_timestamps]) # median

                    #event.trigger.tels_with_trigger.append(tel_id)
                    #event.trigger.tel[tel_id].time = event_time * u.s

            yield event

    @property
    def observation_blocks(self):
        return {} # Fill with metadata obs Id to obs container, valid for obs run
    
    @property
    def scheduling_blocks(self):
        return {} # Fill with metadata sche ID to sched container, valid for obs run
    
    @property
    def simulation_block(self):
        return {}

    @property
    def simulation_config(self) -> Dict[int, SimulationConfigContainer]:
        return self._simulation_config

    @property
    def datalevels(self) -> list[DataLevel]:
        return self.datalevel

    @property
    def obs_ids(self) -> Iterable[int]:
        # ToCheck: will this be compatible in the future, e.g. with merged MC files
        return self._observation_blocks.keys()
    
    @property
    def subarray(self) -> SubarrayDescription:
        return self._subarray_info



class PanoReconstructor(Reconstructor):
    def __call__(self, event: ArrayEventContainer):
        event.dl2.geometry["PanoReconstructor"] = ReconstructedGeometryContainer()