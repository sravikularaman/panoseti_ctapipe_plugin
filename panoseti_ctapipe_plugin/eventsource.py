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

        #-----To modify for two-telescope array-----data? all_timestamps? metadata?
        
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

            for tel_id, raw_event in enumerate(data, start=1):
                event.r0.tel[tel_id].waveform = np.array(raw_event).T[np.newaxis, :, np.newaxis] #ctapipe wants (n_channels, n_pixels, n_samples) and raw_event is (n_pixels) which is an element from data which is (n_samples, n_pixels)

            #Timestamp median of 4 quabo timestamps
            for tel_id, quabo_timestamps in enumerate(all_timestamps, start=1):
                # Each quabo_timestamps = list of 4 timestamp arrays (for 4 quabos)
                if all(len(ts) > i for ts in quabo_timestamps):
                    event_time = np.median([ts[i] for ts in quabo_timestamps])
                    event.trig.tels_with_trigger.append(tel_id)
                    event.trig.tel[tel_id].time = event_time * u.s

            yield event

        
