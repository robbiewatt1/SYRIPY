import torch
import torch.fft as fft
import numpy as np
from Wavefront import Wavefront

# Remove these after checks
from FieldSolver import EdgeRadSolver
from Track import Track
import matplotlib.pyplot as plt

c_light = 0.29979245


class OpticalElement:
    """
    Base class for optical elements. All elements should override the propagate
    function
    """

    def propagate(self, wavefront):
        pass


class RayleighSommerfeldProp(OpticalElement):
    """
    Propagates the wavefront using the Rayleigh–Sommerfeld expression. We
    fourier transform the field, multiply by the transfer function then
    transform back. This is the most accurate propagation method.
    """

    def __init__(self, z):
        """
        :param z: Propagation distance.
        """
        self.z = z

    def propagate(self, wavefront, pad=None):
        """
        Propogates wavefront.
        :param wavefront: Wavefront to be propagated.
        :param pad: Int setting padding amount. Default is none
        """

        if pad:
            wavefront.pad_wavefront(pad)

        wave_k = wavefront.omega / c_light
        lambd = 2.0 * torch.pi / wave_k
        fx = fft.fftshift(fft.fftfreq(wavefront.n_samples_xy[0],
                                      d=wavefront.delta[0],
                                      device=wavefront.device))
        fy = fft.fftshift(fft.fftfreq(wavefront.n_samples_xy[1],
                                      d=wavefront.delta[1],
                                      device=wavefront.device))
        fx, fy = torch.meshgrid(fx, fy, indexing="ij")
        tran_func = torch.exp(1j * 2. * torch.pi * self.z
                              * (1. / lambd**2. - fx**2. - fy**2.)**0.5)
        fig, ax = plt.subplots()
        ax.pcolormesh(((1. - lambd**2 * fx**2. - lambd**2 *
        fy**2.)**0.5).cpu())
        plt.show()
        field = wavefront.field.reshape(wavefront.n_samples_xy[0],
                                        wavefront.n_samples_xy[1], 2)
        field = fft.fft2(fft.fftshift(field, dim=(0, 1)), dim=(0, 1))
        new_field = fft.ifftshift(fft.ifft2(field * tran_func[:, :, None],
                                            dim=(0, 1)), dim=(0, 1))
        wavefront.field = new_field.flatten(0, 1)
        wavefront.z = wavefront.z + self.z


class FresnelProp(OpticalElement):

    def __init__(self, z):
        """
        :param z: Propagation distance.
        """
        self.z = z

    def propagate(self, wavefront, pad=None):
        """
        Propogates wavefront.
        :param wavefront: Wavefront to be propagated.
        :param pad: Int setting padding amount. Default is none
        """

        if pad:
            wavefront.pad_wavefront(pad)

        wave_k = wavefront.omega / c_light
        lambd = 2.0 * torch.pi / wave_k
        fx = fft.fftshift(fft.fftfreq(wavefront.n_samples_xy[0],
                                      d=wavefront.delta[0],
                                      device=wavefront.device))
        fy = fft.fftshift(fft.fftfreq(wavefront.n_samples_xy[1],
                                      d=wavefront.delta[1],
                                      device=wavefront.device))
        fx, fy = torch.meshgrid(fx, fy, indexing="ij")

        tran_func = torch.exp(-1j * torch.pi * lambd * self.z
                              * (fx**2. + fy**2))
        tran_func = torch.exp(torch.tensor(1j * wave_k * self.z)) * tran_func
        field = wavefront.field.reshape(wavefront.n_samples_xy[0],
                                        wavefront.n_samples_xy[1], 2)
        field = fft.fft2(fft.fftshift(field, dim=(0, 1)), dim=(0, 1))
        new_field = fft.ifftshift(fft.ifft2(field * tran_func[:, :, None],
                                            dim=(0, 1)), dim=(0, 1))
        wavefront.field = new_field.flatten(0, 1)
        wavefront.z = wavefront.z + self.z


class FraunhoferProp(OpticalElement):
    """
    Propagates the wavefront by some distance using the Fraunhofer method.
    This can also be used to propagate to the fourier plane of a lens when
    the Fresnel approximation holds.
    """

    def __init__(self, z):
        """
        :param z: Propagation distance (focal length is propagating to lens
                  focus)
        """
        self.z = z

    def propagate(self, wavefront, pad=None):
        """
        Propogates wavefront.
        :param wavefront: Wavefront to be propagated.
        :param pad: Int setting padding amount. Default is none
        """

        if pad:
            wavefront.pad_wavefront(pad)

        wave_k = wavefront.omega / c_light
        lambd = 2.0 * torch.pi / wave_k

        # Get new axes
        deltaX_i = (wavefront.wf_bounds[1] - wavefront.wf_bounds[0]) \
                   / wavefront.n_samples_xy[0]
        deltaY_i = (wavefront.wf_bounds[3] - wavefront.wf_bounds[2]) \
                   / wavefront.n_samples_xy[1]
        prop_bounds = [-0.5 * lambd * self.z / deltaX_i,
                       0.5 * lambd * self.z / deltaX_i,
                       -0.5 * lambd * self.z / deltaY_i,
                       0.5 * lambd * self.z / deltaY_i]
        x_new = torch.linspace(prop_bounds[0], prop_bounds[1],
                               wavefront.n_samples_xy[0],
                               device=wavefront.device)
        y_new = torch.linspace(prop_bounds[2], prop_bounds[3],
                               wavefront.n_samples_xy[1],
                               device=wavefront.device)
        x_new_mesh, y_new_mesh = torch.meshgrid(x_new, y_new, indexing="ij")

        # Calculate new field
        c = 1 / (1j * lambd * self.z) \
            * torch.exp(1j * wave_k / (2. * self.z) * (x_new_mesh**2. +
                                                       y_new_mesh**2.))
        field = wavefront.field.reshape(wavefront.n_samples_xy[0],
                                        wavefront.n_samples_xy[1], 2)
        new_field = c[:, :, None] * torch.fft.ifftshift(torch.fft.fft2(
            torch.fft.fftshift(field, dim=(0, 1)), dim=(0, 1)), dim=(0, 1))

        # Update bounds / field
        wavefront.field = new_field.flatten(0, 1)
        wavefront.z = wavefront.z + self.z
        wavefront.update_bounds(prop_bounds)


class DebyeProp(OpticalElement):
    """
    Propagates the wavefront to the focus of a lens using the Debye-wolf method.
    The field extent cannot be larger than a circle of radius focal_length.
    """

    def __init__(self, focal_length, aper_radius=None):
        """
        :param focal_length: Focal length of lens.
        :param aper_radius: Radius of aperture before lens.
        """
        if aper_radius and aper_radius > focal_length:
            raise Exception("aper_radius must be smaller than focal length")

        self.focal_length = focal_length
        if not aper_radius:
            self.num_aper = focal_length

    def propagate(self, wavefront):
        """
        Propagates wavefront. Aperture is applied first to avoid complex
        wave-vector
        :param wavefront: Wavefront to be propagated.
        """

        k0 = wavefront.omega / c_light
        lambd = 2.0 * torch.pi / k0
        kx = k0 * wavefront.x_axis / self.focal_length
        ky = k0 * wavefront.y_axis / self.focal_length
        kx, ky = torch.meshgrid(kx, ky, indexing="ij")
        kr = (kx**2. + ky**2.)**0.5
        # The abs here avoids complex wave-vectors. Field should be zeros
        # when k0^2 - kr^2 is negative so shouldn't matter
        kz = torch.abs(k0**2. - kr**2.)**0.5

        # Form conv kernels
        G_x = (k0 / kz)**0.5 * (k0 * ky**2. + kz * kx**2.) / (k0 * kr**2.)
        G_y = (k0 / kz)**0.5 * (kz - k0) * kx * kz / (k0 * kr**2.)
        G_z = -(k0 / kz)**0.5 * kx / k0
        G_tx = torch.stack((G_x, G_y, G_z)).permute(1, 2, 0)
        G_ty = torch.stack((-1*G_y, G_x, G_z)).permute(1, 2, 0)
        field = wavefront.field.reshape(wavefront.n_samples_xy[0],
                                        wavefront.n_samples_xy[1], 2)

        # TODO Might need to do some roations for y
        # TODO Need to add * torch.exp(1j * kz * self.focal_length) *


        # Do x pol component
        field_x = fft.ifftshift(fft.ifft2(G_tx * field[:, :, 0, None],
                                          dim=(0, 1)), dim=(0, 1))
        field_y = fft.ifftshift(fft.ifft2(G_ty * field[:, :, 1, None],
                                          dim=(0, 1)), dim=(0, 1))

        wavefront.field = (field_y + field_x).flatten(0, 1)
        Wavefront.dims = 3
        wavefront.z = wavefront.z + self.focal_length


class ThinLens(OpticalElement):

    def __init__(self, focal_length):
        self.focal_length = focal_length

    def propagate(self, wavefront):
        tf = torch.exp(-1j * wavefront.omega / (2 * self.focal_length * c_light)
                * (wavefront.coords[:, 0]**2.0 + wavefront.coords[:, 1]**2.0))
        wavefront.field = wavefront.field * tf


class CircularAperture(OpticalElement):

    def __init__(self, radius):
        self.radius = radius

    def propagate(self, wavefront):
        r = (wavefront.coords[:, 0]**2.0 + wavefront.coords[:, 1]**2.0)**0.5
        mask = torch.where(r < self.radius, 1, 0)[:, None]
        print(r.get_device(), mask.device)
        wavefront.field = wavefront.field * mask



if __name__ == "__main__":
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    track = Track(device=device)
    track.load_file("./track.npy")
    wavefnt = Wavefront(0, 3.77e6,
                        np.array([-0.03, 0.03, -0.031, 0.031]),
                        np.array([50, 50]), 2, device=device)
    wavefnt.field[:, 0] = 1
    aper = CircularAperture(0.03)
    aper.propagate(wavefnt)
    wavefnt.pad_wavefront(20)
    wavefnt.plot_intensity()
    prop = DebyeProp(0.1)
    prop.propagate(wavefnt)
    wavefnt.plot_intensity()
    plt.show()