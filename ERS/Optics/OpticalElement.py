import torch
from ..Wavefront import Wavefront
from typing import List


class OpticalElement:
    """
    Base class for optical elements. All elements should override the propagate
    function
    """

    def __init__(self):
        self.new_shape = None   # Wavefront Shape after propagation
        self.new_bounds = None  # wavefront bounds after propagation

    def propagate(self, wavefront: Wavefront) -> None:
        pass


class ThinLens(OpticalElement):
    """
    Thin lens class. Multiplies the wavefront with a quadratic phase.
    """

    def __init__(self, focal_length: float) -> None:
        """
        :param focal_length: Focal length of the lens
        """
        super().__init__()
        self.focal_length = focal_length
        self.c_light = 0.29979245

    def propagate(self, wavefront: Wavefront) -> None:
        """
        Propagates the wavefront through the lens.
        :param wavefront: Wavefront to be propagated.
        """
        tf = torch.exp((wavefront.coords[0, :]**2.0
                        + wavefront.coords[1, :]**2.0) * -1j * wavefront.omega
                       / (2. * self.focal_length * self.c_light))
        wavefront.field = wavefront.field * tf


class CircularAperture(OpticalElement):
    """
    Circular aperture class. Just sets field to zero if outside area.
    """
    # TODO add centre shift
    def __init__(self, radius: float) -> None:
        """
        :param radius: The radius of the aperture.
        """
        super().__init__()
        self.radius = radius

    def propagate(self, wavefront: Wavefront) -> None:
        """
        Propagates the wavefront through the aperture.
        :param wavefront: Wavefront to be propagated.
        """
        r = (wavefront.coords[0, :]**2.0 + wavefront.coords[1, :]**2.0)**0.5
        mask = torch.where(r < self.radius, 1, 0)[None, :]
        wavefront.field = wavefront.field * mask


class RectangularAperture(OpticalElement):

    """
    Rectangular aperture class. Just sets field to zero if outside area.
    """
    # TODO add centre shift

    def __init__(self, size: List[float]) -> None:
        """
        :param size: The size of the aperture: [length_x, length_y].
        """
        super().__init__()
        self.size = size

    def propagate(self, wavefront: Wavefront) -> None:
        """
        Propagates the wavefront through the aperture.
        :param wavefront: Wavefront to be propagated.
        """
        r = (wavefront.coords[0, :]**2.0 + wavefront.coords[1, :]**2.0)**0.5
        mask = torch.where(torch.abs(wavefront.coords[0, :]) < self.size[0],
                           1, 0)[None, :]
        mask = torch.where(torch.abs(wavefront.coords[1, :]) < self.size[1],
                           1, 0)[None, :] * mask
        wavefront.field = wavefront.field * mask
