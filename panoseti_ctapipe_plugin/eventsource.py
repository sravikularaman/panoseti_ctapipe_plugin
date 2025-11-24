import numpy as np
import astropy.units as u
import pypff

from ctapipe.io import EventSource, DataLevel
from ctapipe.io.datawriter import ArrayEventContainer
from ctapipe.reco import Reconstructor
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
from .instrument import subarray 

class PanoEventSource(EventSource):
    is_simulation = False
    datalevels = (DataLevel.DL0,)
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
            
            event = ArrayEventContainer()
            event.count = i
            event.trig.tels_with_trigger = []

            for tel_id, data in enumerate(telescope_data, start=1):
                raw_event = data[i]

                event.r0.tel[tel_id].waveform = np.array(raw_event)[np.newaxis, :, np.newaxis]
                # ctapipe wants (n_channels, n_pixels, n_samples) and raw_event is (n_pixels) which is an element from data which is (n_samples, n_pixels)

                # Timestamp across quabos
                quabo_timestamps = all_timestamps_by_telescope[tel_id - 1]
                if all(len(ts) > i for ts in quabo_timestamps):
                    
                    event_time = np.min([ts[i] for ts in quabo_timestamps])
                    # event_time = np.median([ts[i] for ts in quabo_timestamps]) # median

                    event.trig.tels_with_trigger.append(tel_id)
                    event.trig.tel[tel_id].time = event_time * u.s

            yield event

        
