from ctapipe.io import EventSource, DataLevel
from ctapipe.io.datawriter import ArrayEventContainer
import numpy as np
import pypff
from .instrument import subarray  # import the subarray definition

class PanoEventSource(EventSource):
    is_simulation = False
    datalevels = (DataLevel.DL0,)
    subarray = subarray

    @classmethod
    def is_compatible(cls, path):
        return str(path).endswith(".pff")

    def _generator(self):
        pff_file = pypff.file()
        pff_file.open(self.input_url)

        for i, evt in enumerate(pff_file.events):  # adapt to your PFF API
            container = ArrayEventContainer()
            container.count = i
            container.r0.tel[1].waveform = np.array(evt.waveforms, dtype=np.float32)
            container.trig.time = evt.timestamp
            yield container

        pff_file.close()
