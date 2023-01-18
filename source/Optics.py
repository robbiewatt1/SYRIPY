import torch
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


class FraunhoferProp(OpticalElement):

    def __init__(self, distance):
        self.z = distance

    def propagate(self, wavefront, pad=False):

        if pad:
            wavefront.pad_wavefront()

        wave_k = wavefront.omega / c_light
        lambd = 2.0 * torch.pi / wave_k

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
        x_new_mesh, y_new_mesh = torch.meshgrid(x_new, y_new, indexing="xy")
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
class FresnelProp(OpticalElement):

    def __init__(self, distance):
        self.distance = distance

    def propagate(self, wavefront):

        """
        freq_x = torch.fft.fftfreq(wavefront.n_samples_xy[0],
                             d=(wavefront.wf_bounds[1] - wavefront.wf_bounds[0])
                               / wavefront.n_samples_xy[0])
        freq_y = torch.fft.fftfreq(wavefront.n_samples_xy[1],
                             d=(wavefront.wf_bounds[3] - wavefront.wf_bounds[2])
                               / wavefront.n_samples_xy[1])
        freq_x, freq_y = torch.meshgrid(freq_x, freq_y, indexing="xy")
        tf = torch.exp(1j * (wavefront.omega * self.distance / c_light -
                             2. * torch.pi**2.0 * c_light * self.distance
                             * (freq_x**2.0 + freq_y**2.0) / wavefront.omega))
        field_k = torch.fft.fft2(,
            dim=(0, 1))
        wavefront.field = torch.fft.ifft2(field_k * tf[:, :, None],
                                          dim=(0, 1)).flatten(0, 1)
        """


class RayleighSommerfeldProp(OpticalElement):

    def __init__(self, distance):
        self.distance = distance

    def propagate(self, wavefront):
        freq_x = torch.fft.fftfreq(wavefront.n_samples_xy[0],
                             d=(wavefront.wf_bounds[1] - wavefront.wf_bounds[0])
                               / wavefront.n_samples_xy[0])
        freq_y = torch.fft.fftfreq(wavefront.n_samples_xy[1],
                             d=(wavefront.wf_bounds[3] - wavefront.wf_bounds[2])
                               / wavefront.n_samples_xy[1])
        freq_x, freq_y = torch.meshgrid(freq_x, freq_y, indexing="xy")
        tf = torch.exp(1j * (wavefront.omega * self.distance / c_light -
                             2. * torch.pi**2.0 * c_light * self.distance
                             * (freq_x**2.0 + freq_y**2.0) / wavefront.omega))
        field_k = torch.fft.fft2(wavefront.field.reshape(
            wavefront.n_samples_xy[0], wavefront.n_samples_xy[1], 2),
            dim=(0, 1))
        wavefront.field = torch.fft.ifft2(field_k * tf[:, :, None],
                                          dim=(0, 1)).flatten(0, 1)


if __name__ == "__main__":
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    track = Track(device=device)
    track.load_file("./track.npy")
    wavefnt = Wavefront(1.7526625849289021, 3.77e6,
                        np.array([-0.022, 0.022, -0.022, 0.022]),
                        np.array([200, 200]), device=device)
    slvr = EdgeRadSolver(wavefnt, track)
    slvr.auto_res()
    slvr.solve(200)
    #wavefnt.field[:, :] = 1
    aper = CircularAperture(0.02)
    aper.propagate(wavefnt)
    wavefnt.pad_wavefront(3)
    wavefnt.plot_intensity()


    free_space_prop = FraunhoferProp(30)
    free_space_prop.propagate(wavefnt)


    wavefnt.plot_intensity(ds_fact=32)#log_plot=False, axes_lim=[-8e-5, 8e-5,
    # -8e-5,
    # 8e-5])
    plt.show()