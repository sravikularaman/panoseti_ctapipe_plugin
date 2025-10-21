from ctapipe.reco import Reconstructor
from ctapipe.containers import ReconstructedGeometryContainer
from ctapipe.io.datawriter import ArrayEventContainer

class PanoReconstructor(Reconstructor):
    def __call__(self, event: ArrayEventContainer):
        event.dl2.geometry["Panoseti"] = ReconstructedGeometryContainer()
