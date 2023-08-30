import torch
import torch.fft as fft
import numpy as np
from ..Wavefront import Wavefront
from .OpticalElement import OpticalElement
from typing import List, Optional


class FreeSpace(OpticalElement):

    def __init__(self, z: float, pad: int, new_shape: List[int],
                 new_bounds: List[float]) -> None:
        """
        :param z: Propagation distance.
        :param pad: Int setting padding amount. Default is none
        :param new_shape: Gives the shape of the output wavefront if using
         CZT transform [nx, ny]
        :param new_bounds: Gives the bounds of the output wavefront if using
         CTZ transform [x_min, x_max, y_min, y_max].
        """
        super().__init__(pad)
        self.z = z
        self.new_shape = new_shape
        self.new_bounds = new_bounds
        self.c_light = 0.299792458

        if self.new_shape and self.new_bounds:
            self.use_czt = True
        elif bool(self.new_shape) ^ bool(self.new_bounds):
            raise Exception("Both new_shape and new_bounds must be defined to "
                            "use czt")
        else:
            self.use_czt = False

    def propagate(self, wavefront: Wavefront) -> None:
        """
        Propagates wavefront.
        :param wavefront: Wavefront to be propagated.
        """
        super().propagate(wavefront)

    def _chirp_z_1d(self, x: torch.Tensor, m: int, f_lims: List[float],
                    fs: float, endpoint: bool = True, power_2: bool = True
                    ) -> torch.Tensor:
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

        wk2 = torch.exp((-1j * np.pi * scale * k ** 2) / m)
        awk2 = torch.exp(-2j * np.pi * f_lims[0]/fs * k[:n]) * wk2[:n]

        if power_2:
            nfft = int(2 ** np.ceil(np.log(m + n - 1) / np.log(2)))
        else:
            nfft = m + n - 1

        fwk2 = torch.fft.fft(1 / torch.hstack(
            (torch.flip(wk2[1:n], dims=[-1]), wk2[:m])), nfft)
        wk2 = wk2[:m]
        y = torch.fft.ifft(fwk2 * torch.fft.fft(x * awk2, nfft))
        y = y[..., slice(n-1, n+m-1)] * wk2
        return y

    def _chirp_z_2d(self, x: torch.Tensor, m: List[int], f_lims: List[float],
                    fs: List[float], endpoint: bool = True,
                    power_2: bool = True) -> torch.Tensor:
        """
        2D chirp Z transform function. Performs the transform on the last two
        dimensions of x. We apply the 1d function along the inner dimension then
        flip and do the second.
        :param x: 2d signal array.
        :param m: Dimension of output array [mx, my].
        :param f_lims: Frequency limits [fx0, fx1, fy0, fy1]
        :param fs: Total length of output space [fsx, fsy].
        :param endpoint: If true, f_lims[1] / f_lims[3] are included.
        :param power_2: If true, array will be padded before fft to power of 2
        :return y: Transformed result
        """
        y = self._chirp_z_1d(x, m[1], [f_lims[2], f_lims[3]], fs[1], endpoint,
                             power_2)
        y = self._chirp_z_1d(torch.swapaxes(y, -1, -2), m[0],
                             [f_lims[0], f_lims[1]], fs[0], endpoint, power_2)
        return torch.swapaxes(y, -1, -2)

    def _inv_chirp_z_2d(self, x: torch.Tensor, m: List[int],
                        f_lims: List[float], fs: List[float],
                        endpoint: bool = True, power_2: bool = True
                        ) -> torch.Tensor:
        """
        2D inverse chirp Z transform function. Used function above but flips and
         normalises the output
        :param x: 2d signal array.
        :param m: Dimension of output array [mx, my].
        :param f_lims: Frequency limits [fx0, fx1, fy0, fy1]
        :param fs: Total length of output space [fsx, fsy].
        :param endpoint: If true, f_lims[1] / f_lims[3] are included.
        :param power_2: If true, array will be padded before fft to power of 2
        :return y: Transformed result
        """
        f_lims = [-f_lims[0], -f_lims[1], -f_lims[2], -f_lims[3]]
        y = self._chirp_z_2d(x, m, f_lims, fs, endpoint, power_2)
        return y / (x.shape[1] * x.shape[2])


class RayleighSommerfeldProp(FreeSpace):
    """
    Propagates the wavefront using the Rayleigh–Sommerfeld expression. We
    fourier transform the field, multiply by the transfer function then
    transform back. This is the most accurate propagation method.
    """

    def __init__(self, z: float, pad: Optional[int] = None,
                 new_shape: Optional[List[int]] = None,
                 new_bounds: Optional[List[float]] = None) -> None:
        """
        :param z: Propagation distance.
        :param pad: Int setting padding amount. Default is none
        :param new_shape: Gives the shape of the output wavefront if using
         CZT transform [nx, ny]
        :param new_bounds: Gives the bounds of the output wavefront if using
        CTZ transform [x_min, x_max, y_min, y_max].
        """
        super().__init__(z, pad, new_shape, new_bounds)

    def propagate(self, wavefront: Wavefront) -> None:
        """
        Propagates wavefront. Aperture is applied first to avoid complex
        wave-vector. If new_shape / new_bounds is set then a czt is used
        rather than a fourier transform. CZT does not need padding.
        :param wavefront: Wavefront to be propagated.
        """
        super().propagate(wavefront)

        if self.new_shape and self.new_bounds:
            use_czt = True
        elif bool(self.new_shape) ^ bool(self.new_bounds):
            use_czt = False
            print("Both new_shape and new_bounds must be defined to use czt. "
                  "Resorting back to fft.")
        else:
            use_czt = False

        wave_k = wavefront.omega / self.c_light
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

        if use_czt:
            new_field = self._chirp_z_2d(field, self.new_shape, self.new_bounds,
                            [wavefront.wf_bounds[1] - wavefront.wf_bounds[0],
                             wavefront.wf_bounds[3] - wavefront.wf_bounds[1]])
            wavefront.update_bounds(self.new_bounds, self.new_shape)
        else:
            new_field = fft.ifftshift(fft.ifft2(field * tran_func[None, :, :]))
        wavefront.field = new_field.flatten(1, 2)
        wavefront.z = wavefront.z + self.z


class FresnelProp(FreeSpace):

    def __init__(self, z: float, pad: Optional[int] = None,
                 new_shape: Optional[List[int]] = None,
                 new_bounds: Optional[List[float]] = None,
                 fourier_shape: Optional[List[int]] = None,
                 fourier_bounds: Optional[List[float]] = None) -> None:
        """
        :param z: Propagation distance.
        :param pad: Int setting padding amount. Default is none
        :param new_shape: The shape of the output wavefront if using
         CZT transform [nx, ny]
        :param new_bounds: The bounds of the output wavefront if using
        CZT transform [x_min, x_max, y_min, y_max].
        :param fourier_shape: The shape of the fourier plane if using the CZT
         transform for first transform.
        param fourier_bounds: The bounds of the fourier plane if using the CZT
         transform for first transform.
        """
        super().__init__(z, pad, new_shape, new_bounds)
        self.fourier_shape = fourier_shape
        self.fourier_bounds = fourier_bounds

    def propagate(self, wavefront: Wavefront) -> None:
        """
        Propagates wavefront. Aperture is applied first to avoid complex
        wave-vector. If new_shape / new_bounds is set then a czt is used
        rather than a fourier transform. CZT does not need padding.
        :param wavefront: Wavefront to be propagated.
        """
        super().propagate(wavefront)

        if self.fourier_shape and self.fourier_bounds:
            use_czt_f = True
        elif bool(self.fourier_shape) ^ bool(self.fourier_bounds):
            use_czt_f = False
            print("Both new_shape and new_bounds must be defined to use czt. "
                  "Resorting back to fft.")
        else:
            use_czt_f = False

        wave_k = wavefront.omega / self.c_light
        lambd = 2.0 * torch.pi / wave_k
        field = wavefront.field.reshape(2, wavefront.n_samples_xy[0],
                                        wavefront.n_samples_xy[1])

        if use_czt_f:  # Using czt in first transform
            delta_x = 1. / (self.fourier_bounds[1] - self.fourier_bounds[0])
            delta_y = 1. / (self.fourier_bounds[3] - self.fourier_bounds[2])
            field = fft.fftshift(self._chirp_z_2d(
                field, self.fourier_shape, self.fourier_bounds,
                [1. / wavefront.delta[0], 1. / wavefront.delta[1]]))
            fx = fft.fftfreq(self.fourier_shape[0],
                                          d=delta_x, device=wavefront.device)
            fy = fft.fftfreq(self.fourier_shape[1],
                                          d=delta_y, device=wavefront.device)
            fx, fy = torch.meshgrid(fx, fy, indexing="ij")

            field = field * torch.exp(
                -1j * torch.pi * lambd * self.z * (fx**2. + fy**2))[None, :, :]
            wavefront.update_bounds(wavefront.wf_bounds, self.fourier_shape)
        else:  # Just using standard fft
            field = fft.fft2(fft.fftshift(field))
            fx = fft.fftfreq(wavefront.n_samples_xy[0],
                             d=wavefront.delta[0], device=wavefront.device)
            fy = fft.fftfreq(wavefront.n_samples_xy[1],
                             d=wavefront.delta[1], device=wavefront.device)
            fx, fy = torch.meshgrid(fx, fy, indexing="ij")
            field = field * torch.exp(-1j * torch.pi * lambd * self.z
                                          * (fx**2. + fy**2))[None, :, :]
        if self.use_czt:
            old_cx = (wavefront.wf_bounds[1] + wavefront.wf_bounds[0]) / 2
            old_cy = (wavefront.wf_bounds[3] + wavefront.wf_bounds[2]) / 2
            f_lims = [self.new_bounds[0] - old_cx, self.new_bounds[1] - old_cx,
                      self.new_bounds[2] - old_cy, self.new_bounds[3] - old_cy]
            new_field = self._inv_chirp_z_2d(torch.fft.ifftshift(
                field), self.new_shape, f_lims,
                [wavefront.wf_bounds[1] - wavefront.wf_bounds[0],
                 wavefront.wf_bounds[3] - wavefront.wf_bounds[2]])
            wavefront.update_bounds(self.new_bounds, self.new_shape)
        else:
            new_field = fft.ifftshift(fft.ifft2(field))
        wavefront.field = (torch.exp(torch.tensor(1j * wave_k * self.z))
                           * new_field).flatten(1, 2)
        wavefront.z = wavefront.z + self.z


class FraunhoferProp(FreeSpace):
    """
    Propagates the wavefront by some distance using the Fraunhofer method.
    This can also be used to propagate to the fourier plane of a lens when
    the Fresnel approximation holds.
    """

    def __init__(self, z: float, pad: Optional[int] = None,
                 new_shape: Optional[List[int]] = None,
                 new_bounds: Optional[List[float]] = None) -> None:
        """
        :param z: Propagation distance (focal length is propagating to lens
                  focus)
        :param pad: Int setting padding amount. Default is none
        :param new_shape: Gives the shape of the output wavefront if using
         CZT transform [nx, ny]
        :param new_bounds: Gives the bounds of the output wavefront if using
        CTZ transform [x_min, x_max, y_min, y_max].
        """
        super().__init__(z, pad, new_shape, new_bounds)

    def propagate(self, wavefront: Wavefront) -> None:
        """
        Propagates wavefront.
        :param wavefront: Wavefront to be propagated.
        """
        super().propagate(wavefront)

        wave_k = wavefront.omega / self.c_light
        lambd = 2.0 * torch.pi / wave_k
        delta_x = (wavefront.wf_bounds[1] - wavefront.wf_bounds[0]) \
                  / wavefront.n_samples_xy[0]
        delta_y = (wavefront.wf_bounds[3] - wavefront.wf_bounds[2])\
                  / wavefront.n_samples_xy[1]
        full_out_size = [lambd * self.z / delta_x, lambd * self.z / delta_y]

        if self.use_czt:
            new_bounds = self.new_bounds
            new_shape = self.new_shape
        else:
            new_bounds = [-0.5 * full_out_size[0], 0.5 * full_out_size[0],
                          -0.5 * full_out_size[1], 0.5 * full_out_size[1]]
            new_shape = wavefront.n_samples_xy

        field = wavefront.field.reshape(2, wavefront.n_samples_xy[0],
                                        wavefront.n_samples_xy[1])
        wavefront.update_bounds(new_bounds, new_shape)
        c = 1. / (1j * lambd * self.z) \
            * torch.exp(1j * wave_k / (2. * self.z)
                        * (wavefront.x_array**2. + wavefront.y_array**2.))

        if self.use_czt:
            new_field = self._chirp_z_2d(
                field, self.new_shape, self.new_bounds, full_out_size)\
                        * c[None, :, :] * delta_x * delta_y
        else:
            new_field = fft.ifftshift(fft.fft2(torch.fft.fftshift(field)))\
                        * c[None, :, :] * delta_x * delta_y
        wavefront.z = wavefront.z + self.z
        wavefront.field = new_field.flatten(1, 2)


class FresnelPropQS(FreeSpace):
    """
    Propagates the wavefront by some distance using the Fresnel method.
    The quadratic phase term is subtracted from the wavefront before
    propagation.
    """

    def __init__(self, z: float, pad: Optional[int] = None,
                 new_shape: Optional[List[int]] = None,
                 new_bounds: Optional[List[float]] = None,
                 input_upsample: Optional[int] = None) -> None:
        """
        :param z: Propagation distance.
        :param pad: Int setting padding amount. Default is none
        :param new_shape: The shape of the output wavefront if using change of
        shape.
        :param new_bounds: The bounds of the output wavefront if using change
        of shape.
        :param input_upsample: Factor to up-sample input wavefront by. This is
        done using bilinear interpolation.
        """

        super().__init__(z, pad, new_shape, new_bounds)
        self.input_upsample = input_upsample

    def propagate(self, wavefront: Wavefront) -> None:
        """
        Propagates wavefront. Aperture is applied first to avoid complex
        wave-vector. If new_shape / new_bounds is set then a czt is used
        rather than a fourier transform. CZT does not need padding.
        :param wavefront: Wavefront to be propagated.
        """
        super().propagate(wavefront)

        wave_k = wavefront.omega / self.c_light
        lambd = 2.0 * torch.pi / wave_k

        # Remove quadratic phase term
        wavefront.field = wavefront.field / torch.exp(1j * wave_k * (
                (wavefront.coords[0, :] - wavefront.source_location[0])**2.0
              + (wavefront.coords[1, :] - wavefront.source_location[1])**2.0)
                                                      / (2 * wavefront.curv_r))
        if self.input_upsample:
            wavefront.interpolate_wavefront(fact=self.input_upsample)

        field_scaled = wavefront.field.reshape(2, wavefront.n_samples_xy[0],
                                               wavefront.n_samples_xy[1])

        # Do convolution on reduced part
        field_scaled = fft.fft2(fft.fftshift(field_scaled))
        fx = fft.fftfreq(wavefront.n_samples_xy[0],
                         d=wavefront.delta[0], device=wavefront.device)
        fy = fft.fftfreq(wavefront.n_samples_xy[1],
                         d=wavefront.delta[1], device=wavefront.device)

        fx, fy = torch.meshgrid(fx, fy, indexing="ij")
        c = (wavefront.curv_r / (wavefront.curv_r + self.z))
        kernel = torch.exp(-1j * torch.pi * lambd * self.z * c
                           * (fx**2. + fy**2.))
        new_field = c * fft.ifftshift(fft.ifft2(field_scaled * kernel))
        wavefront.field = new_field.flatten(1, 2)

        # TODO this needs to also include the source location
        # Update wavefront bounds
        bounds = [(wavefront.wf_bounds[0] * (wavefront.curv_r + self.z)
                   - wavefront.source_location[0] * self.z) / wavefront.curv_r,
                  (wavefront.wf_bounds[1] * (wavefront.curv_r + self.z)
                   - wavefront.source_location[0] * self.z) / wavefront.curv_r,
                  (wavefront.wf_bounds[2] * (wavefront.curv_r + self.z)
                   - wavefront.source_location[1] * self.z) / wavefront.curv_r,
                  (wavefront.wf_bounds[3] * (wavefront.curv_r + self.z)
                   - wavefront.source_location[1] * self.z) / wavefront.curv_r]


        #full_out_size = [(wavefront.wf_bounds[1] - wavefront.wf_bounds[0])
        #                 * (wavefront.curv_r + self.z) / wavefront.curv_r,
        #                 (wavefront.wf_bounds[3] - wavefront.wf_bounds[2])
        #                 * (wavefront.curv_r + self.z) / wavefront.curv_r]
        #bounds = [-0.5 * full_out_size[0], 0.5 * full_out_size[0],
        #          -0.5 * full_out_size[1], 0.5 * full_out_size[1]]
        shape = [wavefront.n_samples_xy[0], wavefront.n_samples_xy[1]]
        wavefront.update_bounds(bounds, shape)

        if self.new_shape and self.new_bounds:
            new_axes = [torch.linspace(self.new_bounds[0], self.new_bounds[1],
                                    self.new_shape[0], device=wavefront.device),
                        torch.linspace(self.new_bounds[2], self.new_bounds[3],
                                    self.new_shape[1], device=wavefront.device)]
            wavefront.interpolate_wavefront(new_axes=new_axes)

        # Add analytical part back in
        wavefront.field = wavefront.field * torch.exp(1j * wave_k * (
                (wavefront.coords[0, :] - wavefront.source_location[0])**2.0
              + (wavefront.coords[1, :] - wavefront.source_location[1])**2.0)
                            / (2. * (wavefront.curv_r + self.z)))
        wavefront.field = torch.exp(torch.tensor(1j * wave_k * self.z)) \
                          * wavefront.field
        wavefront.z = wavefront.z + self.z


class FraunhoferPropQS(FreeSpace):
    """
    Propagates the wavefront by some distance using the Fraunhofer method.
    The quadratic phase term is subtracted from the wavefront before
    propagation.
    """

    def __init__(self, z: float, pad: Optional[int] = None,
                 new_shape: Optional[List[int]] = None,
                 new_bounds: Optional[List[float]] = None,
                 input_upsample: Optional[int] = None) -> None:
        """
        :param z: Propagation distance.
        :param pad: Int setting padding amount. Default is none
        :param new_shape: The shape of the output wavefront if using change of
        shape.
        :param new_bounds: The bounds of the output wavefront if using change
        of shape.
        :param input_upsample: Factor to up-sample input wavefront by. This is
        done using bilinear interpolation.
        """

        super().__init__(z, pad, new_shape, new_bounds)
        self.input_upsample = input_upsample

    def propagate(self, wavefront: Wavefront) -> None:
        super().propagate(wavefront)

        wave_k = wavefront.omega / self.c_light
        lambd = 2.0 * torch.pi / wave_k

        # Remove quadratic phase term
        wavefront.field = wavefront.field / torch.exp(1j * wave_k * (
                (wavefront.coords[0, :] - wavefront.source_location[0])**2.0
              + (wavefront.coords[1, :] - wavefront.source_location[1])**2.0)
                                                      / (2 * wavefront.curv_r))

        if self.input_upsample:
            wavefront.interpolate_wavefront(fact=self.input_upsample)

        field_scaled = wavefront.field.reshape(2, wavefront.n_samples_xy[0],
                                               wavefront.n_samples_xy[1])

        # Do convolution on reduced part
        field_scaled = fft.fft2(fft.fftshift(field_scaled))
        fx = fft.fftfreq(wavefront.n_samples_xy[0],
                         d=wavefront.delta[0], device=wavefront.device)
        fy = fft.fftfreq(wavefront.n_samples_xy[1],
                         d=wavefront.delta[1], device=wavefront.device)

        fx, fy = torch.meshgrid(fx, fy, indexing="ij")
        kernel = torch.exp(-1j * torch.pi * lambd * wavefront.curv_r
                           * (fx**2. + fy**2.)) * wavefront.curv_r / self.z
        new_field = fft.ifftshift(fft.ifft2(field_scaled * kernel))
        wavefront.field = new_field.flatten(1, 2)

        # update wavefront bounds
        bounds = [(wavefront.wf_bounds[0] - wavefront.source_location[0])
                  * (self.z / wavefront.curv_r),
                  (wavefront.wf_bounds[1] - wavefront.source_location[0])
                  * (self.z / wavefront.curv_r),
                  (wavefront.wf_bounds[2] - wavefront.source_location[1])
                  * (self.z / wavefront.curv_r),
                  (wavefront.wf_bounds[3] - wavefront.source_location[1])
                  * (self.z / wavefront.curv_r)]
        shape = [wavefront.n_samples_xy[0], wavefront.n_samples_xy[1]]
        wavefront.update_bounds(bounds, shape)

        if self.new_shape and self.new_bounds:
            new_axes = [torch.linspace(self.new_bounds[0], self.new_bounds[1],
                                    self.new_shape[0], device=wavefront.device),
                        torch.linspace(self.new_bounds[2], self.new_bounds[3],
                                    self.new_shape[1], device=wavefront.device)]
            wavefront.interpolate_wavefront(new_axes=new_axes)

        # Add analytical part back in
        zrz = (self.z - wavefront.curv_r) / self.z
        wavefront.field = wavefront.field * torch.exp(1j * wave_k * (
                wavefront.coords[0, :] * (zrz * wavefront.coords[0, :]
                                          - wavefront.source_location[0])
                + wavefront.coords[1, :] * (zrz * wavefront.coords[1, :]
                                            - wavefront.source_location[1]))
                            / (2. * self.z))
        wavefront.field = torch.exp(torch.tensor(1j * wave_k * self.z)) \
                          * wavefront.field

        wavefront.z = wavefront.z + self.z


'''
class DebyeProp(FreeSpace):
    """
    Propagates the wavefront to the focus of a lens using the Debye-wolf method.
    The field extent cannot be larger than a circle of radius focal_length.
    """

    def __init__(self, z, pad=None, new_shape=None, new_bounds=None,
                 aper_radius=None, calc_z=False):
        """
        :param z: Propagation distance (focal length is propagating to lens
                  focus)
        :param pad: Int setting padding amount. Default is none
        :param new_shape: Gives the shape of the output wavefront if using
         CZT transform [nx, ny]
        :param new_bounds: Gives the bounds of the output wavefront if using
        CTZ transform [xmin, xmax, ymin, ymax].
        :param aper_radius: Radius of aperture. Used to clip wavenumbers that
         are complex.
        :param calc_z: If true, the z component of the field will also be
        calculated (usually small).
        """
        super().__init__(z, pad, new_shape, new_bounds)
        self.num_aper = aper_radius
        self.calc_z = calc_z

        if aper_radius and aper_radius > z:
            raise Exception("aper_radius must be smaller than focal length")

        if not aper_radius:
            self.num_aper = z

    def propagate(self, wavefront):
        """
        Propagates wavefront. Aperture is applied first to avoid complex
        wave-vector. If new_shape / new_bounds is set then a czt is used
        rather than a fourier transform. CZT does not need padding.
        :param wavefront: Wavefront to be propagated.
        """

        if self.new_shape and self.new_bounds:
            use_czt = True
        elif bool(self.new_shape) ^ bool(self.new_bounds):
            use_czt = False
            print("Both new_shape and new_bounds must be defined to use czt. "
                  "Resorting back to fft.")
        else:
            use_czt = False

        k0 = wavefront.omega / c_light
        lambd = 2.0 * torch.pi / k0
        # First sort the new bounds
        delta_x = (wavefront.wf_bounds[1] - wavefront.wf_bounds[0])\
                  / wavefront.n_samples_xy[0]
        delta_y = (wavefront.wf_bounds[3] - wavefront.wf_bounds[2])\
                  / wavefront.n_samples_xy[1]
        full_out_size = [lambd * self.z / delta_x, lambd * self.z / delta_y]
        if not use_czt:
            new_bounds = [-0.5 * full_out_size[0], 0.5 * full_out_size[0],
                          -0.5 * full_out_size[1], 0.5 * full_out_size[1]]
            new_shape = wavefront.n_samples_xy

        kx = k0 * wavefront.x_array / self.z
        ky = k0 * wavefront.y_array / self.z
        kr = (kx**2. + ky**2.)**0.5

        # The abs here avoids complex wave-vectors. Field should be zeros
        # when k0^2 - kr^2 is negative so shouldn't matter
        kz = torch.abs(k0**2. - kr**2.)**0.5

        # Form conv kernels
        g_x = (k0 / kz)**0.5 * (k0 * ky**2. + kz * kx**2.) / (k0 * kr**2.)
        g_y = (k0 / kz)**0.5 * (kz - k0) * kx * kz / (k0 * kr**2.)
        g_z = -(k0 / kz)**0.5 * kx / k0
        g_tx = torch.exp(1j * kz * self.z) * torch.stack((g_x, g_y, g_z))
        g_ty = torch.exp(1j * kz * self.z) * torch.stack((-1 * g_y, g_x, g_z))
        field = wavefront.field.reshape(2, wavefront.n_samples_xy[0],
                                        wavefront.n_samples_xy[1])

        # TODO Might need to do some rotations for y
        if use_czt:
            field_x = self._chirp_z_2d(g_tx * field[0, :, :], self.new_shape,
                                       self.new_bounds, full_out_size)
            field_y = self._chirp_z_2d(g_ty * field[1, :, :], self.new_shape,
                                       self.new_bounds, full_out_size)
        else:
            field_x = fft.ifftshift(fft.ifft2(
                torch.fft.fftshift(g_tx * field[0, :, :])))
            field_y = fft.ifftshift(fft.ifft2(
                torch.fft.fftshift(g_ty * field[1, :, :])))

        wavefront.z = wavefront.z + self.z
        wavefront.update_bounds(self.new_bounds, self.new_shape)
        if self.calc_z:
            Wavefront.dims = 3
            wavefront.field = (field_x + field_y).flatten(1, 2)
        else:
            wavefront.field = (field_x + field_y).flatten(1, 2)[[0, 1]]
'''
