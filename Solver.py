import numpy as np
import matplotlib.pyplot as plt
from scipy import interpolate
import cProfile
from Track import Track
from Wavefront import Wavefront
import time
import os
import torch

c_light = 0.29979245


class EdgeRadSolver:
    """
    Class which solves the Liénard–Wiechert field at a wavefront for a given
    particle trajectory.
    """

    def __init__(self, wavefront, track):
        """
        :param wavefront: Instance of Wavefront class
        :param track: Instance of Track class
        """
        self.wavefront = wavefront
        self.track = track

    def auto_res(self, sample_x=10, ds_windows=10, ds_min=3):
        """
        Automatically down-samples the track based on large values of
        objective function obj = |grad(1/grad(g))|. Where g(t) is the phase
        function.
        :param sample_x: Number of test sample points in x
        :param ds_windows: Number of windows in which the path is down sampled
        :param ds_min: log10 of minimum down sampling factor
        """
        # TODO: Make auto_res work with torch

        init_samples = self.track.time.shape[0]

        # Calculate grad(1/ grad(phi)) at sample_x points.
        test_index = int(self.wavefront.x_axis.shape[0] / sample_x)
        r_obs = np.stack([self.wavefront.x_axis[::test_index],
                          np.zeros(sample_x),
                          np.ones(sample_x) * self.wavefront.z]).T[:, :, None] \
                - self.track.r
        r_norm = np.linalg.norm(r_obs, axis=1)
        phase_samples = self.track.time[None, :] + r_norm / c_light
        phase_grad = np.gradient(phase_samples, self.track.time, axis=1)
        grad_inv_grad = np.gradient(1 / phase_grad, self.track.time, axis=1)
        obj_val = np.max(np.abs(grad_inv_grad), axis=0)

        # Calculate mean objective value in windows
        window_shape = (-1, int(obj_val.shape[0] / ds_windows))
        obj_val_windows = np.mean(obj_val.reshape(window_shape), axis=1)

        # Calculate the log10 down-sampling factor for each window
        window_ds_fact = -np.log10(obj_val_windows
                                   / np.max(obj_val_windows))
        # Clip down sampling at min value
        window_ds_fact = np.clip(window_ds_fact, a_min=0, a_max=ds_min)
        window_ds_fact = window_ds_fact

        # Apply down-sampling to track values
        # time
        time_list = list(self.track.time.reshape(window_shape))
        time_list = [t[::int(10**window_ds_fact[i])]
                     for i, t in enumerate(time_list)]
        self.track.time = np.concatenate(time_list)

        # Beta
        b_list = list(self.track.beta.reshape((3, *window_shape))
                      .transpose(1, 2, 0))
        beta_list = [b[::int(10**window_ds_fact[i])]
                     for i, b in enumerate(b_list)]
        self.track.beta = np.concatenate(beta_list).T

        # Position
        r_list = list(self.track.r.reshape((3, *window_shape))
                      .transpose(1, 2, 0))
        r_list = [r[::int(10**window_ds_fact[i])]
                  for i, r in enumerate(r_list)]
        self.track.r = np.concatenate(r_list).T
        print("Reduction factor: ", init_samples / self.track.time.shape[0])

    def solve(self, t_start, t_end, n_int):
        """
        Main function to solve the radiation field at the wavefront.
        Interaction limits must be within time array of track
        :param t_start: Stat time of integration (ns)
        :param t_end: End time of
        :param n_int: Number of sample in integration
        """

        # TODO: Make checks work with torch
        """
        # check that t_start and t_end are acceptable
        f_interp = interpolate.interp1d(self.track.time, self.track.r[2])
        if t_start < self.track.time[0] or t_start > self.track.time[-1]:
            raise Exception(f"Integration boundaries must be within provided "
                            f"track. Track boundaries are"
                            f" {self.track.time[0]}  and {self.track.time[-1]}")
        if f_interp(t_end) > self.wavefront.z:
            raise Exception(f"Particle is beyond wavefront z at t_end."
                            f" z pos at time {t_end} is {f_interp(t_end)}")
        """

        # Set integration points
        start_index = torch.argmin(torch.abs(self.track.time - t_start))
        end_index = torch.argmin(torch.abs(self.track.time - t_end))
        skip = int((end_index - start_index) / n_int)+1
        t_int_points = self.track.time[start_index:end_index:skip]
        # Calculate observation points
        r_obs = self.wavefront.coords[..., None] - self.track.r
        r_norm = torch.linalg.norm(r_obs, dim=1, keepdim=True)
        n_dir = r_obs / r_norm

        # Calculate the phase function and gradient
        phase = self.track.time + r_norm / c_light
        phase = phase - phase[..., 0, None]
        phase_grad = torch.unsqueeze(torch.gradient(torch.squeeze(phase),
                                                    spacing=(self.track.time,),
                                                    dim=1)[0], dim=1)

        # Now calculate integrand samples
        int1_samples = (self.track.beta - n_dir) / (r_norm * phase_grad)
        int2_samples = c_light * n_dir / (self.wavefront.omega * r_norm ** 2.0
                                          * phase_grad)
        # Repeat phase along r axis for interpolation
        phase = phase.repeat(1, 3, 1)
        int1_func = SplineInterp(phase, int1_samples)
        int2_func = SplineInterp(phase, int2_samples)

        phase_int = phase[..., start_index:end_index:skip]
        real_part = (self.filon_cos(int1_func, self.wavefront.omega, phase_int)
                     + self.filon_sin(int2_func, self.wavefront.omega,
                                      phase_int))
        imag_part = (self.filon_sin(int1_func, self.wavefront.omega, phase_int)
                     - self.filon_cos(int2_func, self.wavefront.omega,
                                      phase_int))

        self.wavefront.field = (real_part + 1j * imag_part).reshape(
            self.wavefront.n_samples_xy[0], self.wavefront.n_samples_xy[1], 3)



        """
        # loop through wavefront array
        for i, xi in enumerate(self.wavefront.x_axis):
            print(i)
            for j, yj in enumerate(self.wavefront.y_axis):
                r_obs = np.array([xi, yj, self.wavefront.z])[:, None] \
                        - self.track.r
                r_norm = np.linalg.norm(r_obs, axis=0)

                # Calc phase / gradient
                phase_samples = self.track.time + r_norm / c_light
                phase_samples = phase_samples - phase_samples[0]
                phase_grad = np.gradient(phase_samples, self.track.time)

                # Phase function
                phase_func = interpolate.InterpolatedUnivariateSpline(
                    self.track.time, phase_samples, k=2)

                # Form integrand function (f / gp)
                n_dir_samples = r_obs / r_norm
                int1_samples = (self.track.beta - n_dir_samples) \
                    / (r_norm * phase_grad)
                int2_samples = c_light * n_dir_samples \
                    / (self.wavefront.omega * r_norm**2.0
                       * phase_grad)

                int1_func = interpolate.InterpolatedUnivariateSpline(
                    phase_samples, int1_samples, k=2)
                int2_func = interpolate.InterpolatedUnivariateSpline(
                    phase_samples, int2_samples, k=2)
                phase_int = phase_func(t_int_points)


                fig, ax = plt.subplots()
                ax.scatter(t_int_points, phase_int)
                fig, ax = plt.subplots()
                ax.scatter(t_int_points, int1_func(phase_int))
                ax.plot(self.track.time, int1_func(phase_samples))
                fig, ax = plt.subplots()
                ax.plot(self.track.time, int2_func(phase_samples))
                ax.scatter(t_int_points, int2_func(phase_int))
                fig, ax = plt.subplots()
                plt.scatter(self.track.time, 1/ phase_grad)
                plt.show()


                # Perform real and imaginary integrals
                real_part = self.filon_cos(int1_func, self.wavefront.omega,
                                           phase_int) + \
                            self.filon_sin(int2_func, self.wavefront.omega,
                                           phase_int)
                imag_part = self.filon_sin(int1_func, self.wavefront.omega,
                                           phase_int) - \
                            self.filon_cos(int2_func, self.wavefront.omega,
                                           phase_int)
                # need to add phase part from integration variable change
                centre = (real_part + 1j * imag_part) * np.exp(1j *
                                self.wavefront.omega * phase_samples[0])
                # Get the edge term
                start_edge_real, start_edge_imag = self.solve_edge(
                    self.track.time[0], self.track.time[0],
                    self.track.beta[:, 0], r_obs[:, 0],
                    self.wavefront.omega)
                end_edge_real, end_edge_imag = self.solve_edge(
                    self.track.time[end_index], self.track.time[end_index],
                    self.track.beta[:, end_index], r_obs[:, end_index],
                    self.wavefront.omega)
                edge = (start_edge_real[index] - end_edge_real[index]
                        + (start_edge_imag[index] - end_edge_imag[index])
                        * 1j)

                self.wavefront.field[index, i, j] = centre# + edge
        """

    @staticmethod
    def solve_edge(t, t_0, beta_0, r_0, omega):
        """
        Calculates the integral of the end parts in free space.
        :param t: Current time
        :param t_0: Time when beta and r are defined
        :param beta_0: Velocity of particle at t=0
        :param r_0: Position of particle at t=0
        :param omega: Oscillation frequency
        :return: Real and imaginary parts (real, imag)
        """
        b2 = (beta_0**2).sum()
        rb = (r_0 * beta_0).sum()
        r2 = (r_0**2.0).sum()
        r_n = r2**0.5
        n = r_0 / r_n
        phase = t + r_n / c_light
        t = t - t_0

        # Calculate phase gradient
        r_grad = (b2 * t + rb) / r_n
        n_grad = (beta_0 * r_n - r_grad * r_0) / r2
        phase_grad = 1 + r_grad / c_light
        phase_grad_2nd = (b2 * r2 - rb**2) / (c_light * r_n**3)
        func_real = (beta_0 - n) / r_n
        func_imag = - n * c_light / (omega * r_n**2.0)

        func_grad_real = -beta_0 * r_grad / r2 - (n_grad * r_n - r_grad * n) \
                         / r2
        func_grad_imag = - c_light * (n_grad * r2 - 2 * r_n * r_grad * n) \
                    / (r2**2 * omega)

        real_part = (func_grad_real * phase_grad - func_real * phase_grad_2nd) \
                    / (omega**2.0 * phase_grad**3) + func_imag / (omega *
                                                                  phase_grad)
        imag_part = (func_grad_imag * phase_grad - func_imag * phase_grad_2nd) \
                    / (omega**2.0 * phase_grad**3) - func_real / (omega *
                                                                  phase_grad)
        return (real_part * np.cos(omega * phase)
                - imag_part * np.sin(omega * phase),
                imag_part * np.cos(omega * phase)
                + real_part * np.sin(omega * phase))

    def filon_sin(self, func, omega, x_samples):
        """
        Filon based method for integrating function multiplied by a rapidly
        oscillating sine wave. I = int[f(x) sin(omega x)], omega >> 1. Uses a
        quadratic approximation for f(x), allowing I to be solved analytically.
        :param func: Function to be integrated, f(x).
        :param omega: Oscillation frequency.
        :param x_samples: Sample points of integration.
        :return: Integration result.
        """
        lower_lim = x_samples[..., :-1]
        upper_lim = x_samples[..., 1:]
        mid_point = lower_lim + (upper_lim - lower_lim) / 2
        int_points = torch.stack([lower_lim, mid_point, upper_lim], dim=-1)
        f = func(int_points.flatten(-2, -1)).reshape(self.wavefront.n_samples,
                                                     3, -1, 3)
        A = torch.stack([torch.ones_like(int_points), int_points,
                         int_points**2.0], dim=-1)
        coeffs = torch.linalg.solve(A, f)
        moment_0th = self.sin_moment(omega, int_points[..., ::2])
        moment_1st = self.x_sin_moment(omega, int_points[..., ::2])
        moment_2nd = self.x2_sin_moment(omega, int_points[..., ::2])
        return torch.sum(coeffs[..., 0] * moment_0th
                         + coeffs[..., 1] * moment_1st
                         + coeffs[..., 2] * moment_2nd, dim=-1)

    def filon_cos(self, func, omega, x_samples):
        """
        Filon based method for integrating function multiplied by a rapidly
        oscillating cosine wave. I = int[f(x) cos(omega x)], omega >> 1. Uses a
        quadratic approximation for f(x), allowing I to be solved analytically.
        :param func: Function to be integrated, f(x).
        :param omega: Oscillation frequency.
        :param x_samples: Sample points of integration.
        :return: Integration result.
        """
        # Create sample point array
        lower_lim = x_samples[..., :-1]
        upper_lim = x_samples[..., 1:]
        mid_point = lower_lim + (upper_lim - lower_lim) / 2
        int_points = torch.stack([lower_lim, mid_point, upper_lim], dim=-1)
        f = func(int_points.flatten(-2, -1)).reshape(self.wavefront.n_samples,
                                                     3, -1, 3)
        A = torch.stack([torch.ones_like(int_points), int_points,
                         int_points**2.0], dim=-1)
        coeffs = torch.linalg.solve(A, f)
        moment_0th = self.cos_moment(omega, int_points[..., ::2])
        moment_1st = self.x_cos_moment(omega, int_points[..., ::2])
        moment_2nd = self.x2_cos_moment(omega, int_points[..., ::2])
        return torch.sum(coeffs[..., 0] * moment_0th
                         + coeffs[..., 1] * moment_1st
                         + coeffs[..., 2] * moment_2nd, dim=-1)

    @staticmethod
    def sin_moment(omega, x_lim):
        """
        :param omega: Oscillation frequency
        :param x_lim: Integration limits [xi, xf]
        :return: Zeroth moment [sin(omega x)]
        """
        return (torch.cos(omega * x_lim[..., 0])
                - torch.cos(omega * x_lim[..., 1])) / omega

    @staticmethod
    def x_sin_moment(omega, x_lim):
        """
        :param omega: Oscillation frequency
        :param x_lim: Integration limits [xi, xf]
        :return: First moment [x sin(omega x)]
        """
        return (torch.sin(omega * x_lim[..., 1]) - x_lim[..., 1] * omega
                * torch.cos(omega * x_lim[..., 1])
                - torch.sin(omega * x_lim[..., 0]) + x_lim[..., 0] * omega
                * torch.cos(omega * x_lim[..., 0])) / omega**2

    @staticmethod
    def x2_sin_moment(omega, x_lim):
        """
        :param omega: Oscillation frequency
        :param x_lim: Integration limits [xi, xf]
        :return: Second moment [x^2 sin(omega x)]
        """
        return ((2 - x_lim[..., 1]**2 * omega**2)
                * torch.cos(omega * x_lim[..., 1]) + 2 * omega * x_lim[..., 1]
                * torch.sin(omega * x_lim[..., 1])
                - (2 - x_lim[..., 0]**2 * omega**2)
                * torch.cos(omega * x_lim[..., 0]) - 2 * omega * x_lim[..., 0]
                * torch.sin(omega * x_lim[..., 0])) / omega**3.0

    @staticmethod
    def cos_moment(omega, x_lim):
        """
        :param omega: Oscillation frequency
        :param x_lim: Integration limits [xi, xf]
        :return: Zeroth moment [cos(omega x)]
        """
        return (torch.sin(omega * x_lim[..., 1])
                - torch.sin(omega * x_lim[..., 0])) / omega

    @staticmethod
    def x_cos_moment(omega, x_lim):
        """
        :param omega: Oscillation frequency
        :param x_lim: Integration limits [xi, xf]
        :return: First moment [x cos(omega x)]
        """
        return (torch.cos(omega * x_lim[..., 1]) + x_lim[..., 1] * omega
                * torch.sin(omega * x_lim[..., 1])
                - torch.cos(omega * x_lim[..., 0]) - x_lim[..., 0] * omega
                * torch.sin(omega * x_lim[..., 0])) / omega**2

    @staticmethod
    def x2_cos_moment(omega, x_lim):
        """
        :param omega: Oscillation frequency
        :param x_lim: Integration limits [xi, xf]
        :return: Second moment [x^2 cos(omega x)]
        """
        return ((x_lim[..., 1]**2 * omega**2 - 2)
                * torch.sin(omega * x_lim[..., 1]) + 2 * omega * x_lim[..., 1]
                * torch.cos(omega * x_lim[..., 1])
                - (x_lim[..., 0]**2 * omega**2 - 2)
                * torch.sin(omega * x_lim[..., 0]) - 2 * omega * x_lim[..., 0]
                * torch.cos(omega * x_lim[..., 0])) / omega**3.0


class SplineInterp:
    """
    Cubic spline interpolator class using pytorch. Can perform multiple
    interpolations along batch dimension.
    """

    def __init__(self, x, y):
        """
        :param x: x samples
        :param y: y samples
        """
        self.x = x
        self.y = y
        self.device = x.device
        self.A = torch.tensor([
            [1, 0, -3, 2],
            [0, 1, -2, 1],
            [0, 0, 3, -2],
            [0, 0, -1, 1]],
            dtype=x.dtype, device=self.device)

    def h_poly(self, x):
        """
        Calculates the Hermite polynomials
        :param x: Locations to calculate polynomials at
        :return: Hermite polynomials
        """
        xx = x[..., None, :] ** torch.arange(
            4, device=self.device)[..., :, None]
        return torch.matmul(self.A, xx)

    def __call__(self, xs):
        """
        Performs interpolation at location xs.
        :param xs: locations to interpolate
        :return: Interpolated value
        """
        m = ((self.y[..., 1:] - self.y[..., :-1]) /
             (self.x[..., 1:] - self.x[..., :-1]))
        m = torch.cat([m[..., [0]], (m[..., 1:] + m[..., :-1]) / 2,
                       m[..., [-1]]], dim=-1)
        idx = torch.searchsorted(self.x[..., 1:].contiguous(), xs)
        dx = (torch.gather(self.x, dim=-1, index=idx+1)
              - torch.gather(self.x, dim=-1, index=idx))
        hh = self.h_poly((xs - torch.gather(self.x, dim=-1, index=idx)) / dx)
        return (hh[..., 0, :] * torch.gather(self.y, dim=-1, index=idx)
                + hh[..., 1, :] * torch.gather(m, dim=-1, index=idx)
                * dx + hh[..., 2, :] * torch.gather(self.y, dim=-1, index=idx+1)
                + hh[..., 3, :] * torch.gather(m, dim=-1, index=idx+1) * dx)


if __name__ == "__main__":
    print(print(torch.__version__))
    print(torch.version.cuda)
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    #device = "cpu"
    print(device)

    track = Track("./track.npy", device=device)
    print(track.time[0], track.time[-1])
    wavefnt = Wavefront(1.7526625849289021, 3.77e6,
                        [-0.01, 0.01, -0.01, 0.01],
                        [50, 50], device=device)

    slvr = EdgeRadSolver(wavefnt, track)
    cProfile.run("slvr.solve(0, 10, 5)")
    wavefnt.plot_intensity()
    plt.show()

