import torch

c_light = 0.29979245


class OpticalElement:
    """
    Base class for optical elements. All elements should override the propagate
    function
    """

    def propagate(self, wavefront):
        pass


class ThinLens(OpticalElement):
    """
    Thin lens class. Multiplies the wavefront with a quadratic phase.
    """

    def __init__(self, focal_length):
        """
        :param focal_length: Focal length of the lens
        """
        self.focal_length = focal_length

    def propagate(self, wavefront):
        """
        Propagates the wavefront through the lens.
        :param wavefront: Wavefront to be propagated.
        """
        tf = torch.exp(-1j * wavefront.omega / (2 * self.focal_length * c_light)
                * (wavefront.coords[0, :]**2.0 + wavefront.coords[1, :]**2.0))
        wavefront.field = wavefront.field * tf


class CircularAperture(OpticalElement):
    """
    Circular aperture class. Just sets field to zero if outside area.
    """
    # TODO add centre shift
    def __init__(self, radius):
        """
        :param radius: The radius of the aperture.
        """
        self.radius = radius

    def propagate(self, wavefront):
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

    def __init__(self, size):
        """
        :param size: The size of the aperture: [length_x, length_y].
        """
        self.size = size

    def propagate(self, wavefront):
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
