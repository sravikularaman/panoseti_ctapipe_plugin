from ctapipe.io import EventSource, DataLevel
from ctapipe.io.datawriter import ArrayEventContainer
import numpy as np
import astropy.units as u
import pypff
from .instrument import subarray 

class PanoEventSource(EventSource):
    is_simulation = False
    datalevels = (DataLevel.DL0,)
    subarray = subarray

    @classmethod
    def is_compatible(cls, path):
        return str(path).endswith(".pff")

    def _generator(self):
        pff_file = pypff.io.datapff(str(self.input_url))
        data, metadata = pff_file.readpff(metadata=True)

        all_timestamps = []
        for i in range(4):
            meta = metadata[f'quabo_{i}']
            timestamps = meta['tv_sec'] + meta['tv_usec'] * 1e-6
            #timestamps = timestamps[meta['pkt_num'] != 0] How to take into account packet loss?
            all_timestamps.append(timestamps)

        for i, raw_event in enumerate(data): 
            
            event = ArrayEventContainer()
            event.count = i

            tel_id = 1
            event.r0.tel[tel_id].waveform = np.array(raw_event).T[np.newaxis, :, np.newaxis] #ctapipe wants (n_channels, n_pexels, n_samples)

            #Timestamp median of 4 quabo timestamps
            if all(len(ts) > i for ts in all_timestamps):  
                event_time = np.median([ts[i] for ts in all_timestamps])
                event.trig.tels_with_trigger = [tel_id]
                event.trig.time = event_time * u.s

            yield event

        
