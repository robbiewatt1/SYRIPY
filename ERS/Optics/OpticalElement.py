import torch
from ..Wavefront import Wavefront
from typing import List, Optional


class OpticalElement:
    """
    Base class for optical elements. All elements should override the propagate
    function
    """

    def __init__(self, pad: int):
        """
        :param pad: Int setting padding amount along each axis.
        """
        self.pad = pad

    def propagate(self, wavefront: Wavefront) -> None:
        """
        Base class propagating method. Just does some padding if padding exists.
        :param wavefront: Wavefront to be propagated.
        """
        if self.pad is not None:
            wavefront.pad_wavefront(self.pad)


class ThinLens(OpticalElement):
    """
    Thin lens class. Multiplies the wavefront with a quadratic phase.
    """

    def __init__(self, focal_length: float, pad: Optional[int] = None) -> None:
        """
        :param focal_length: Focal length of the lens
        :param pad: Int setting padding amount along each axis.
        """
        super().__init__(pad)
        self.focal_length = focal_length
        self.c_light = 0.29979245

    def propagate(self, wavefront: Wavefront) -> None:
        """
        Propagates the wavefront through the lens.
        :param wavefront: Wavefront to be propagated.
        """
        super().propagate(wavefront)
        tf = torch.exp((wavefront.coords[0, :]**2.0
                        + wavefront.coords[1, :]**2.0) * -1j * wavefront.omega
                       / (2. * self.focal_length * self.c_light))
        wavefront.field = wavefront.field * tf

        # Update wavefront curvature
        c = self.focal_length / (self.focal_length - wavefront.curv_r)
        wavefront.source_location[0] = wavefront.source_location[0] * c
        wavefront.source_location[1] = wavefront.source_location[0] * c
        wavefront.curv_r = wavefront.curv_r * c


class CircularAperture(OpticalElement):
    """
    Circular aperture class. Just sets field to zero if outside area.
    """
    # TODO add centre shift
    def __init__(self, radius: float, pad: Optional[int] = None) -> None:
        """
        :param radius: The radius of the aperture.
        :param pad: Int setting padding amount along each axis.
        """
        super().__init__(pad)
        self.radius = radius

    def propagate(self, wavefront: Wavefront) -> None:
        """
        Propagates the wavefront through the aperture.
        :param wavefront: Wavefront to be propagated.
        """
        super().propagate(wavefront)
        r = (wavefront.coords[0, :]**2.0 + wavefront.coords[1, :]**2.0)**0.5
        mask = torch.where(r < self.radius, 1, 0)[None, :]
        wavefront.field = wavefront.field * mask


class RectangularAperture(OpticalElement):

    """
    Rectangular aperture class. Just sets field to zero if outside area.
    """
    # TODO add centre shift

    def __init__(self, size: List[float], pad: Optional[int] = None) -> None:
        """
        :param size: The size of the aperture: [length_x, length_y].
        """
        super().__init__(pad)
        self.size = size

    def propagate(self, wavefront: Wavefront) -> None:
        """
        Propagates the wavefront through the aperture.
        :param wavefront: Wavefront to be propagated.
        """
        super().propagate(wavefront)
        mask = torch.where(torch.abs(wavefront.coords[0, :]) < self.size[0],
                           1, 0)[None, :]
        mask = torch.where(torch.abs(wavefront.coords[1, :]) < self.size[1],
                           1, 0)[None, :] * mask
        wavefront.field = wavefront.field * mask
