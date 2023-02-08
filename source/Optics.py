import torch
import torch.fft as fft
import numpy as np
import cmath
from Wavefront import Wavefront

# Remove these after checks
from FieldSolver import EdgeRadSolver
from Track import Track
import matplotlib.pyplot as plt
import matplotlib


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
        field = wavefront.field.reshape(2, wavefront.n_samples_xy[0],
                                        wavefront.n_samples_xy[1])
        field = fft.fft2(fft.fftshift(field))
        new_field = fft.ifftshift(fft.ifft2(field * tran_func[None, :, :]))
        wavefront.field = new_field.flatten(1, 2)
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
        field = wavefront.field.reshape(2, wavefront.n_samples_xy[0],
                                        wavefront.n_samples_xy[1])
        field = fft.fft2(fft.fftshift(field))
        new_field = fft.ifftshift(fft.ifft2(field * tran_func[None, :, :]))
        wavefront.field = new_field.flatten(1, 2)
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

    def propagate(self, wavefront, pad=None, new_shape=None,
                  new_bounds=None):
        """
        Propogates wavefront.
        :param wavefront: Wavefront to be propagated.
        :param pad: Int setting padding amount. Default is none
        :param new_shape: Gives the shape of the output wavefront if using
        CZT transform [nx, ny]
        :param new_bounds: Gives the bounds of the output wavefront if using
        CTZ transform [xmin, xmax, ymin, ymax].
        """

        if pad:
            wavefront.pad_wavefront(pad)

        if new_shape and new_bounds:
            use_czt = True
        elif bool(new_shape) ^ bool(new_bounds):
            use_czt = False
            print("Both new_shape and new_bounds must be defined to use czt. "
                  "Resorting back to fft.")
        else:
            use_czt = False

        wave_k = wavefront.omega / c_light
        lambd = 2.0 * torch.pi / wave_k
        delta_x = (wavefront.wf_bounds[1] - wavefront.wf_bounds[0]) \
                  / wavefront.n_samples_xy[0]
        delta_y = (wavefront.wf_bounds[3] - wavefront.wf_bounds[2])\
                  / wavefront.n_samples_xy[1]
        full_out_size = [lambd * self.z / delta_x, lambd * self.z / delta_y]

        if not use_czt:
            new_bounds = [-0.5 * full_out_size[0],
                          0.5 * full_out_size[0],
                          -0.5 * full_out_size[1],
                          0.5 * full_out_size[1]]
            new_shape = wavefront.n_samples_xy

        field = wavefront.field.reshape(2, wavefront.n_samples_xy[0],
                                        wavefront.n_samples_xy[1])
        wavefront.update_bounds(new_bounds, new_shape)
        c = 1. / (1j * lambd * self.z) \
            * torch.exp(1j * wave_k / (2. * self.z)
                        * (wavefront.x_array**2. + wavefront.y_array**2.))

        if use_czt:
            new_field = c[None, :, :] * chirp_z_2d(field, new_shape, new_bounds,
                                                   full_out_size)
        else:
            new_field = c[None, :, :] * fft.ifftshift(fft.fft2(
                torch.fft.fftshift(field)))

        wavefront.z = wavefront.z + self.z
        wavefront.field = new_field.flatten(1, 2)


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

    def propagate(self, wavefront, pad=None, new_shape=None,
                  new_bounds=None, calc_z=False):
        """
        Propagates wavefront. Aperture is applied first to avoid complex
        wave-vector. If new_shape / new_bounds is set then a czt is used
        rather than a fourier transform. CZT does not need padding.
        :param wavefront: Wavefront to be propagated.
        :param pad: Int setting padding amount. Default is none
        :param new_shape: Gives the shape of the output wavefront if using
        CZT transform [nx, ny]
        :param new_bounds: Gives the bounds of the output wavefront if using
        CTZ transform [xmin, xmax, ymin, ymax].
        :param calc_z: If true, the z component of the field will also be
        calculated (usually small).
        """

        if new_shape and new_bounds:
            use_czt = True
        elif bool(new_shape) ^ bool(new_bounds):
            use_czt = False
            print("Both new_shape and new_bounds must be defined to use czt. "
                  "Resorting back to fft.")
        else:
            use_czt = False

        k0 = wavefront.omega / c_light
        lambd = 2.0 * torch.pi / k0
        # First sort the new bounds
        delta_x = (wavefront.wf_bounds[1] - wavefront.wf_bounds[0]) \
                  / wavefront.n_samples_xy[0]
        delta_y = (wavefront.wf_bounds[3] - wavefront.wf_bounds[2])\
                  / wavefront.n_samples_xy[1]
        full_out_size = [lambd * self.focal_length / delta_x,
                         lambd * self.focal_length / delta_y]
        if not use_czt:
            new_bounds = [-0.5 * full_out_size[0],
                          0.5 * full_out_size[0],
                          -0.5 * full_out_size[1],
                          0.5 * full_out_size[1]]
            new_shape = wavefront.n_samples_xy

        kx = k0 * wavefront.x_array / self.focal_length
        ky = k0 * wavefront.y_array / self.focal_length
        kr = (kx**2. + ky**2.)**0.5

        # The abs here avoids complex wave-vectors. Field should be zeros
        # when k0^2 - kr^2 is negative so shouldn't matter
        kz = torch.abs(k0**2. - kr**2.)**0.5

        # Form conv kernels
        g_x = (k0 / kz)**0.5 * (k0 * ky**2. + kz * kx**2.) / (k0 * kr**2.)
        g_y = (k0 / kz)**0.5 * (kz - k0) * kx * kz / (k0 * kr**2.)
        g_z = -(k0 / kz)**0.5 * kx / k0
        g_tx = torch.exp(1j * kz * self.focal_length) \
               * torch.stack((g_x, g_y, g_z))
        g_ty = torch.exp(1j * kz * self.focal_length) \
               * torch.stack((-1*g_y, g_x, g_z))
        field = wavefront.field.reshape(2, wavefront.n_samples_xy[0],
                                        wavefront.n_samples_xy[1])
        # TODO Might need to do some rotations for y
        fig, ax = plt.subplots()
        pcol = ax.pcolormesh(torch.angle(g_tx[0]))
        fig.colorbar(pcol)
        fig, ax = plt.subplots()
        pcol = ax.pcolormesh(torch.angle(g_tx[1]))
        fig.colorbar(pcol)
        fig, ax = plt.subplots()
        pcol = ax.pcolormesh(torch.angle(g_tx[2]))
        fig.colorbar(pcol)
        plt.show()

        if use_czt:
            field_x = chirp_z_2d(g_tx * field[0, :, :], new_shape,
                                 new_bounds, full_out_size)
            field_y = chirp_z_2d(g_ty * field[1, :, :], new_shape,
                                 new_bounds, full_out_size)
        else:
            field_x = fft.ifftshift(fft.ifft2(
                torch.fft.fftshift(g_tx * field[0, :, :])))
            field_y = fft.ifftshift(fft.ifft2(
                torch.fft.fftshift(g_ty * field[1, :, :])))

        wavefront.z = wavefront.z + self.focal_length
        wavefront.update_bounds(new_bounds, new_shape)
        if calc_z:
            Wavefront.dims = 3
            wavefront.field = (field_x + field_y).flatten(1, 2)
        else:
            wavefront.field = (field_x + field_y).flatten(1, 2)[[0, 1]]


class ThinLens(OpticalElement):

    def __init__(self, focal_length):
        self.focal_length = focal_length

    def propagate(self, wavefront):
        tf = torch.exp(-1j * wavefront.omega / (2 * self.focal_length * c_light)
                * (wavefront.coords[0, :]**2.0 + wavefront.coords[1, :]**2.0))
        wavefront.field = wavefront.field * tf


class CircularAperture(OpticalElement):

    def __init__(self, radius):
        self.radius = radius

    def propagate(self, wavefront):
        r = (wavefront.coords[0, :]**2.0 + wavefront.coords[1, :]**2.0)**0.5
        mask = torch.where(r < self.radius, 1, 0)[None, :]
        wavefront.field = wavefront.field * mask


def chirp_z_1d(x, m, f_lims, fs, endpoint=True, power_2=True):
    """
    1D chirp Z transform function. Generalisation of a DFT that can have
    different sampling in the input and output space. Good for focusing
    simulations where the radiation is concentrated on a small area.
    This is just copied from scipy but implemented using torch.
    :param x: Input signal array. If multidimensional, the last
    :param m: Dimension of output array.
    :param f_lims: Frequency limits [f0, f1].
    :param fs: Total length of output space.
    :param endpoint: If true, f_lims[1] is included.
    :param power_2: If true, array will be padded before fft to power of 2
    for maximum efficiency
    :return y: Transformed result
    """

    # out full dim
    n = x.shape[-1]
    k = torch.arange(max(m, n), device=x.device)

    if endpoint:
        scale = ((f_lims[1] - f_lims[0]) * m) / (fs * (m - 1))
    else:
        scale = (f_lims[1] - f_lims[0]) / fs

    wk2 = torch.exp(-(1j * np.pi * scale * k ** 2) / m)
    awk2 = torch.exp(-2j * np.pi * f_lims[0]/fs * k[:n]) * wk2[:n]

    if power_2:
        nfft = int(2 ** np.ceil(np.log(m + n - 1) / np.log(2)))
    else:
        nfft = m + n - 1

    fwk2 = torch.fft.fft(1 / torch.hstack(
        (torch.flip(wk2[1:n], dims=[-1]), wk2[:m])), nfft)
    wk2 = wk2[:m]
    y = torch.fft.ifft(fwk2 * torch.fft.fft(x * awk2, nfft))
    y = y[..., slice(n - 1, n + m - 1)] * wk2
    return y


def chirp_z_2d(x, m, f_lims, fs, endpoint=True, power_2=True):
    """
    2D chirp Z transform function. Performs the transform on the last two
    dimensions of x. We apply the 1d function along the inner dimension then
    flip and do the second.
    :param x: 2d signal array.
    :param m: Dimension of output array [mx, my].
    :param f_lims: Frequency limits [fx0, fx1, fy0, fy1]
    :param fs: Total length of output space [fsx, fsy].
    :param endpoint: If true, f_lims[1] / fimits[3] are included.
    :param power_2: If true, array will be padded before fft to power of 2
    :return y: Transformed result
    """

    y = chirp_z_1d(x, m[1], [f_lims[2], f_lims[3]], fs[1], endpoint, power_2)
    y = chirp_z_1d(torch.swapaxes(y, -1, -2), m[0], [f_lims[0], f_lims[1]],
                   fs[0], endpoint, power_2)
    return torch.swapaxes(y, -1, -2)


if __name__ == "__main__":

    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    track = Track(device=device)
    track.load_file("./track.npy")

    wavefnt = Wavefront(1.7526625849289021, 3.77e6,
                        [-0.01, 0.01, -0.01, 0.01],
                        [4000, 4000], device=device)

    slvr = EdgeRadSolver(wavefnt, track, device=device)

    slvr.set_dt(1000, flat_power=0.5)
    slvr.solve(400)


    aper = CircularAperture(0.01)
    aper.propagate(wavefnt)
    wavefnt.plot_intensity(ds_fact=4)

    prop = FraunhoferProp(0.105)

    prop.propagate(wavefnt, new_shape=[1000, 1000], new_bounds=[-0.0018, 0.0018,
                                                                -0.002, 0.002])

    wavefnt.plot_intensity(log_plot=True)
    plt.show()

    """
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    wavefnt = Wavefront(0, 3.77e6,
                        np.array([-0.0375, 0.0375, -0.0375, 0.0375]),
                        np.array([7040, 7000]), 2, device=device)

    in_wf = np.load("/Users/rwatt/Documents/Edge-Radiation/SRW/Compressor/SRW_WF0.1.npy")
    in_wf = torch.tensor(in_wf).permute((2, 1, 0)).flatten(1, 2)

    wavefnt.field = in_wf / 1e17
    aper = CircularAperture(0.0375)
    aper.propagate(wavefnt)
    wavefnt.plot_intensity(ds_fact=10)

    prop = FraunhoferProp(0.105)

    prop.propagate(wavefnt, new_shape=[1000, 1000], new_bounds=[-0.0018, 0.0018,
                                                                -0.002, 0.002])
    wavefnt.plot_intensity()
    plt.show()
    """
