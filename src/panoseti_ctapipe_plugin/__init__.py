import numpy as np
import pypff
from typing import Dict, Iterable
from pathlib import Path

import astropy.units as u
from astropy.coordinates import EarthLocation
from astropy.time import Time

from ctapipe.containers import (
    ObservationBlockContainer,
    ReconstructedGeometryContainer,
    SchedulingBlockContainer,
    TriggerContainer,
    ArrayEventContainer,
    SimulationConfigContainer,
    CoordinateFrameType,
    SchedulingBlockType,
    ObservingMode,
    PointingMode,
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

#--------------------------------------CAMERA AND OPTICS DESCRIPTION--------------------------------------

nx, ny = 32, 32
pix_size = 10.0 / 32 * u.deg


geometry = CameraGeometry.make_rectangular(
    npix_x=32, npix_y=32,
    range_x=(-0.04, 0.04),
    range_y=(-0.04, 0.04)
)

readout = CameraReadout(name='panoseti', n_pixels=1024, n_channels=1, reference_pulse_shape=np.array([[]]), reference_pulse_sample_width=1.0*u.ns, sampling_rate=1.0*u.GHz, n_samples=1, n_samples_long=1)

camera = CameraDescription(name="Panoseti", geometry=geometry, readout=readout)

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


#--------------------------------TELESCOPE AND SUBARRAY DESCRIPTION--------------------------------------

telescope_1 = TelescopeDescription(name="Gattini", optics=optics, camera=camera)

telescope_2 = TelescopeDescription(name="Winter", optics=optics, camera=camera)

telescope_3 = TelescopeDescription(name="Fern", optics=optics, camera=camera)

telescope_4 = TelescopeDescription(name="PTI-Heli", optics=optics, camera=camera)

subarray = SubarrayDescription(
    name="Panoseti-Palomar",
    tel_descriptions={1: telescope_1, 2: telescope_2, 3: telescope_3, 4: telescope_4},
    tel_positions={
        1: [177.58, -333.33, 42.82] * u.m,   # Gattini (CORSIKA coordinates)
        2: [-220.15, 33.67, 49.17] * u.m,    # Winter (CORSIKA coordinates)
        3: [-130.43, 190.40, 34.66] * u.m,   # Fern (CORSIKA coordinates)
        4: [-1.0, 97.18, 39.70] * u.m,       # PTI-Heli (CORSIKA coordinates)
    },
    reference_location = EarthLocation(
    lat=33.3564 * u.deg,
    lon=-116.865 * u.deg,
    height=1712 * u.m
)
)

# Mapping of module number to telescope ID
MODULE_TO_TEL_ID = {
    254: 1,  # Gattini
    253: 2,  # Winter
    252: 3,  # Fern
    250: 4,  # PTI-Heli
}

#--------------------------------------EVENT SOURCE------------------------------------------------------


class PanoEventSource(EventSource):
    is_simulation = False
    datalevels = (DataLevel.DL1_IMAGES,) # Images not waveforms

    def __init__(self, input_url=None, subarray_desc=None, sb_type=None, observing_mode=None, 
                 pointing_mode=None, **kwargs):
        """
        Initialize PanoEventSource
        
        Parameters
        ----------
        input_url : str or Path, optional
            Path to the observation run folder containing module_*.pff files and hk.pff
        subarray_desc : SubarrayDescription, optional
            Subarray to use. If None, defaults to the full Panoseti array
        sb_type : SchedulingBlockType, optional
            Type of scheduling block. Defaults to OBSERVATION
        observing_mode : ObservingMode, optional
            Observing mode. Defaults to ON_OFF
        pointing_mode : PointingMode, optional
            Pointing mode. Defaults to TRACK
        **kwargs
            Additional arguments passed to EventSource
        """
        if subarray_desc is None:
            subarray_desc = subarray
        self._subarray = subarray_desc
        
        # Set scheduling block parameters with defaults
        self.sb_type = sb_type if sb_type is not None else SchedulingBlockType.OBSERVATION
        self.observing_mode = observing_mode if observing_mode is not None else ObservingMode.ON_OFF
        self.pointing_mode = pointing_mode if pointing_mode is not None else PointingMode.TRACK
        
        super().__init__(input_url=input_url, **kwargs)
        self._pff_files = []

    @property
    def subarray(self):
        """Obtain the subarray from the EventSource"""
        return self._subarray

    @classmethod
    def is_compatible(cls, path):
        """Check if path is an observation run folder containing .pff files"""
        path = Path(path)
        if not path.is_dir():
            return False
        # Check if there are any module_*.pff files in the directory
        pff_files = list(path.glob("start*ph1024*module_*.*.pff"))
        return len(pff_files) > 0

    def _generator(self):
        """Generator that yields events from the observation run folder"""
        
        # Discover all .pff files in the observation folder
        obs_dir = Path(self.input_url)
        module_files_list = sorted(obs_dir.glob("start*ph1024*module_*.*.pff"))
        
        if not module_files_list:
            raise FileNotFoundError(f"No module_*.pff files found in {obs_dir}")
        
        # Map discovered module files to telescope IDs
        module_files = {}
        for file_path in module_files_list:
            # Extract module number from filename (e.g., "module_254" from filename)
            filename = file_path.name
            parts = filename.split('module_')
            if len(parts) > 1:
                module_num = int(parts[1].split('.')[0])
                
                # Map module number to telescope ID
                if module_num in MODULE_TO_TEL_ID:
                    tel_id = MODULE_TO_TEL_ID[module_num]
                    module_files[tel_id] = str(file_path)
        
        if not module_files:
            raise FileNotFoundError(f"No recognized modules found in {obs_dir}")
        
        # Get sorted telescope IDs
        tel_ids = sorted(module_files.keys())
        
        # Filter by allowed_tels if specified
        if self.allowed_tels is not None:
            module_files = {tid: path for tid, path in module_files.items() 
                          if tid in self.allowed_tels}
            tel_ids = [tid for tid in tel_ids if tid in self.allowed_tels]

        # Read all telescopes' data and metadata
        telescope_data = {}
        telescope_metadata = {}

        for tel_id, file_path in module_files.items():
            pff_file = pypff.io.datapff(file_path)
            data, metadata = pff_file.readpff(metadata=True)
            telescope_data[tel_id] = data
            telescope_metadata[tel_id] = metadata
            self._pff_files.append(pff_file)

        # Collect timestamps and filter packet losses for each telescope
        all_timestamps_by_telescope = {}
        valid_event_mask_by_telescope = {} 

        for tel_id in tel_ids:
            metadata = telescope_metadata[tel_id]
            quabo_timestamps = []
            quabo_valid = []

            for i in range(4):
                meta = metadata[f'quabo_{i}']
                timestamps = meta['tv_sec'] + meta['tv_usec'] * 1e-6

                # Filter out lost packets (pkt_num != 0 means valid data)
                valid_pkt = meta['pkt_num'] != 0
                
                quabo_timestamps.append(timestamps)
                quabo_valid.append(valid_pkt)

            all_timestamps_by_telescope[tel_id] = quabo_timestamps

            valid_events = np.all(np.column_stack(quabo_valid), axis=1)
            valid_event_mask_by_telescope[tel_id] = valid_events
       
        # Loop through events
        num_events = len(telescope_data[min(telescope_data.keys(), key=lambda x: len(telescope_data[x]))])

        event_count = 0
        for i in range(num_events):

            if not all(valid_event_mask_by_telescope[tel_id][i] for tel_id in tel_ids):
                continue
            
            # Check if we've reached max_events limit
            if self.max_events is not None and event_count >= self.max_events:
                break
            
            # Get timestamp from first valid quabo of first telescope
            first_tel_id = tel_ids[0]
            quabo_timestamps = all_timestamps_by_telescope[first_tel_id]
            event_time = np.min([ts[i] for ts in quabo_timestamps if len(ts) > i])
            trigger = TriggerContainer(time=Time(event_time, format='unix'))
            event = ArrayEventContainer(trigger=trigger)
            event.count = event_count
            event.trigger.tels_with_trigger = []

            for tel_id in tel_ids:
                raw_event = telescope_data[tel_id][i]

                event.r0.tel[tel_id].waveform = np.array(raw_event)[np.newaxis, :, np.newaxis]
                # ctapipe wants (n_channels, n_pixels, n_samples) and raw_event is (n_pixels)

                # Timestamp across quabos for this telescope
                quabo_timestamps = all_timestamps_by_telescope[tel_id]
                if all(len(ts) > i for ts in quabo_timestamps):
                    event_time = np.min([ts[i] for ts in quabo_timestamps])
                    event.trigger.tels_with_trigger.append(tel_id)

            yield event
            event_count += 1

    def close(self):
        """Close all open pypff file handles"""
        for pff_file in self._pff_files:
            if hasattr(pff_file, 'close'):
                pff_file.close()
        self._pff_files.clear()

    @property
    def observation_blocks(self):
        """Extract observation metadata from input folder and housekeeping"""
        try:
            # input_url is now the observation run folder
            data_dir = Path(self.input_url)
            hk_file = data_dir / "hk.pff"
            
            if not hk_file.exists():
                return {}
            
            # Load housekeeping data using hkpff
            hkpff = pypff.io.hkpff(str(hk_file))
            hk = hkpff.readhk()
            
            # Extract pointing info from first available mount
            mount_key = None
            for key in ['MOUNT_GATTINI', 'MOUNT_WINTER', 'MOUNT_FERN']:
                if key in hk:
                    mount_key = key
                    break
            
            if mount_key is None:
                return {}
            
            ra_hours = hk[mount_key]["ra_hours"]
            dec_deg = hk[mount_key]["dec_deg"]
            ra_deg = ra_hours * 15  # Convert hours to degrees
            
            # Extract start time from any module file in the folder
            module_files = list(data_dir.glob("start*ph1024*module_*.*.pff"))
            if not module_files:
                return {}
            
            # Parse filename to get start time
            filename = module_files[0].name
            start_str = filename.split('start_')[1].split('.')[0]
            start_time = Time(start_str, format='isot')
            
            obs_id = 0
            obs_block = ObservationBlockContainer(
                obs_id=obs_id,
                producer_id="Panoseti",
                actual_start_time=start_time,
                subarray_pointing_lon=ra_deg * u.deg,
                subarray_pointing_lat=dec_deg * u.deg,
                subarray_pointing_frame=CoordinateFrameType.ICRS,
            )
            return {obs_id: obs_block}
        except Exception as e:
            # Fallback if parsing fails
            return {}
    
    @property
    def scheduling_blocks(self):
        """Extract scheduling block metadata from observation date"""
        try:
            # Extract start time from any module file in the folder
            data_dir = Path(self.input_url)
            module_files = list(data_dir.glob("start*ph1024*module_*.*.pff"))
            if not module_files:
                return {}
            
            # Parse filename to get start time
            filename = module_files[0].name
            start_str = filename.split('start_')[1].split('.')[0]
            start_time = Time(start_str, format='isot')
            
            # Use date as sb_id (YYYYMMDD format)
            sb_id = np.uint64(int(start_time.strftime('%Y%m%d')))
            
            sb_block = SchedulingBlockContainer(
                sb_id=sb_id,
                producer_id="Panoseti",
                sb_type=self.sb_type,
                observing_mode=self.observing_mode,
                pointing_mode=self.pointing_mode,
            )
            return {sb_id: sb_block}
        except Exception as e:
            # Fallback if parsing fails
            return {}
    
    @property
    def simulation_block(self):
        # To fill in if using simulations in the future
        return None

    @property
    def simulation_config(self) -> Dict[int, SimulationConfigContainer]:
        return {}

    @property
    def obs_ids(self) -> Iterable[int]:
        """Return observation IDs from observation blocks"""
        return self.observation_blocks.keys()


#--------------------------------------RECONSTRUCTION----------------------------------------------------

class PanoReconstructor(Reconstructor):
    """Simple reconstructor for PANOSETI events"""
    
    def __call__(self, event: ArrayEventContainer):
        """Apply reconstruction to event"""
        if "PanoReconstructor" not in event.dl2.geometry:
            event.dl2.geometry["PanoReconstructor"] = ReconstructedGeometryContainer()
        return event