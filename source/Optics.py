import torch
from Wavefront import Wavefront

c_light = 0.29979245

class OpticalElement:
    """
    Base class for optical elements. All elements should override the propagate
    function
    """

    def propagate(self, wavefront):
        pass


class ThinLens(OpticalElement):

    def __init__(self, focal_length):
        self.focal_length = focal_length

    def propagate(self, wavefront):
        tf = torch.exp(-1j * wavefront.omega / (2 * self.focal_length * c_light)
                * (wavefront.coords[:, 0]**2.0 + wavefront.coords[:, 1]**2.0))
        wavefront.field = wavefront.field *


if __name__ == "__main__":
    wavefnt = Wavefront(1.7526625849289021, 3.77e6,
                        [-0.01, 0.01, -0.01, 0.01],
                        [200, 200])
    lens = ThinLens(0.105)
    lens.propagate(wavefnt)