import numpy as np
import matplotlib.pyplot as plt
from scipy import interpolate
import cProfile
from Track import Track
from Wavefront import Wavefront
import time
import os
import torch
from torch.profiler import profile, record_function, ProfilerActivity


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

    def auto_res(self, sample_x=None, ds_windows=200, ds_min=2):
        """
        Automatically down-samples the track based on large values of
        objective function obj = |grad(1/grad(g))|. Where g(t) is the phase
        function.
        :param sample_x: Number of test sample points in x
        :param ds_windows: Number of windows in which the path is down sampled
        :param ds_min: log10 of minimum down sampling factor
        """

        if not sample_x:
            sample_x = self.wavefront.x_axis.shape[0]

        # Store initial samples to print reduction
        init_samples = self.track.time.shape[0]
        # Calculate grad(1/ grad(phi)) at sample_x points.
        test_index = int(self.wavefront.x_axis.shape[0] / sample_x
                         * self.wavefront.n_samples_xy[0])
        r_obs = self.wavefront.coords[::test_index, :, None] - self.track.r
        r_norm = torch.linalg.norm(r_obs, dim=1)
        phase_samples = self.track.time[None, :] + r_norm / c_light
        phase_grad = torch.gradient(phase_samples,
                                    spacing=(self.track.time,), dim=1)[0]
        inv_phase_grad = 1 / phase_grad
        grad_inv_grad = torch.gradient(torch.squeeze(inv_phase_grad),
                                       spacing=(self.track.time,), dim=1)[0]
        print(grad_inv_grad.shape)
        obj_val, _ = torch.max(torch.abs(grad_inv_grad), dim=0)

        # Calculate mean objective value in windows
        window_shape = (-1, int(obj_val.shape[0] / ds_windows))
        obj_val_windows, _ = torch.max(obj_val.reshape(window_shape), dim=1)

        # Calculate the log10 down-sampling factor for each window
        window_ds_fact = -torch.log10(obj_val_windows
                                      / torch.max(obj_val_windows))
        # Clip down sampling at min value
        window_ds_fact = torch.clip(window_ds_fact, min=0, max=ds_min)

        # Apply down-sampling to track values
        # time
        time_list = list(self.track.time.reshape(window_shape))
        time_list = [t[::int(10**window_ds_fact[i])]
                     for i, t in enumerate(time_list)]
        self.track.time = torch.cat(time_list)

        # Beta
        b_list = list(self.track.beta.reshape((3, *window_shape))
                      .permute(1, 2, 0))
        beta_list = [b[::int(10**window_ds_fact[i])]
                     for i, b in enumerate(b_list)]
        self.track.beta = torch.cat(beta_list).T

        # Position
        r_list = list(self.track.r.reshape((3, *window_shape))
                      .permute(1, 2, 0))
        r_list = [r[::int(10**window_ds_fact[i])]
                  for i, r in enumerate(r_list)]
        self.track.r = torch.cat(r_list).T
        print("auto_res reduction factor: ", init_samples /
              self.track.time.shape[0])

    def solve(self):
        """
        Main function to solve the radiation field at the wavefront.
        Interaction limits must be within time array of track
        :param t_start: Stat time of integration (ns)
        :param t_end: End time of
        :param n_int: Number of sample in integration
        """

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
        int1 = (self.track.beta - n_dir) / (r_norm * phase_grad)
        int2 = c_light * n_dir / (self.wavefront.omega * r_norm ** 2.0
                                          * phase_grad)
        # Repeat phase along r axis for interpolation
        real_part = (self.filon_cos(phase, int1, self.wavefront.omega)
                     + self.filon_sin(phase, int2, self.wavefront.omega))
        imag_part = (self.filon_sin(phase, int1, self.wavefront.omega)
                     - self.filon_cos(phase, int2, self.wavefront.omega))

        self.wavefront.field = (real_part + 1j * imag_part).reshape(
            self.wavefront.n_samples_xy[0], self.wavefront.n_samples_xy[1], 3)

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

    def filon_sin(self, x_samples, f_samples, omega):
        """
        Filon based method for integrating function multiplied by a rapidly
        oscillating sine wave. I = int[f(x) sin(omega x)], omega >> 1. Uses a
        quadratic approximation for f(x), allowing I to be solved analytically.
        :param x_samples: Sample points of integration.
        :param f_samples: Samples of non-oscillating function
        :param omega: Oscillation frequency.
        :return: Integration result.
        """
        x0 = x_samples[..., :-2:2]
        x1 = x_samples[..., 1:-1:2]
        x2 = x_samples[..., 2::2]
        j0 = self.sin_moment(omega, x0, x2)
        j1 = self.x_sin_moment(omega, x0, x2)
        j2 = self.x2_sin_moment(omega, x0, x2)
        f0 = f_samples[..., :-2:2]
        f1 = f_samples[..., 1:-1:2]
        f2 = f_samples[..., 2::2]
        w0 = (x1 * x2 * j0 - x1 * j1 - x2 * j1 + j2) \
            / ((x0 - x1) * (x0 - x2))
        w1 = (x0 * x2 * j0 - x0 * j1 - x2 * j1 + j2) \
            / ((x1 - x0) * (x1 - x2))
        w2 = (x0 * (j1 - x1 * j0) + x1 * j1 - j2) \
            / ((x0 - x2) * (x2 - x1))
        return torch.sum(w0 * f0 + w1 * f1 + w2 * f2, dim=-1)

    def filon_cos(self, x_samples, f_samples, omega):
        """
        Filon based method for integrating function multiplied by a rapidly
        oscillating cosine wave. I = int[f(x) cos(omega x)], omega >> 1. Uses a
        quadratic approximation for f(x), allowing I to be solved analytically.
        :param x_samples: Sample points of integration.
        :param f_samples: Samples of non-oscillating function
        :param omega: Oscillation frequency.
        :return: Integration result.
        """
        x0 = x_samples[..., :-2:2]
        x1 = x_samples[..., 1:-1:2]
        x2 = x_samples[..., 2::2]
        j0 = self.cos_moment(omega, x0, x2)
        j1 = self.x_cos_moment(omega, x0, x2)
        j2 = self.x2_cos_moment(omega, x0, x2)
        f0 = f_samples[..., :-2:2]
        f1 = f_samples[..., 1:-1:2]
        f2 = f_samples[..., 2::2]
        w0 = (x1 * x2 * j0 - x1 * j1 - x2 * j1 + j2) \
            / ((x0 - x1) * (x0 - x2))
        w1 = (x0 * x2 * j0 - x0 * j1 - x2 * j1 + j2) \
            / ((x1 - x0) * (x1 - x2))
        w2 = (x0 * (j1 - x1 * j0) + x1 * j1 - j2) \
            / ((x0 - x2) * (x2 - x1))
        return torch.sum(w0 * f0 + w1 * f1 + w2 * f2, dim=-1)

    @staticmethod
    def sin_moment(omega, x_low, x_high):
        """
        :param omega: Oscillation frequency
        :param x_low: Lower integration limit
        :param x_high Upper integration limit
        :return: Zeroth moment [sin(omega x)]
        """
        return (torch.cos(omega * x_low)
                - torch.cos(omega * x_high)) / omega

    @staticmethod
    def x_sin_moment(omega, x_low, x_high):
        """
        :param omega: Oscillation frequency
        :param x_low: Lower integration limit
        :param x_high Upper integration limit
        :return: First moment [x sin(omega x)]
        """
        return (torch.sin(omega * x_high) - x_high * omega
                * torch.cos(omega * x_high)
                - torch.sin(omega * x_low) + x_low * omega
                * torch.cos(omega * x_low)) / omega**2

    @staticmethod
    def x2_sin_moment(omega, x_low, x_high):
        """
        :param omega: Oscillation frequency
        :param x_low: Lower integration limit
        :param x_high Upper integration limit
        :return: Second moment [x^2 sin(omega x)]
        """
        return ((2 - x_high**2 * omega**2)
                * torch.cos(omega * x_high) + 2 * omega * x_high
                * torch.sin(omega * x_high)
                - (2 - x_low**2 * omega**2)
                * torch.cos(omega * x_low) - 2 * omega * x_low
                * torch.sin(omega * x_low)) / omega**3.0

    @staticmethod
    def cos_moment(omega, x_low, x_high):
        """
        :param omega: Oscillation frequency
        :param x_low: Lower integration limit
        :param x_high Upper integration limit
        :return: Zeroth moment [cos(omega x)]
        """
        return (torch.sin(omega * x_high)
                - torch.sin(omega * x_low)) / omega

    @staticmethod
    def x_cos_moment(omega, x_low, x_high):
        """
        :param omega: Oscillation frequency
        :param x_low: Lower integration limit
        :param x_high Upper integration limit
        :return: First moment [x cos(omega x)]
        """
        return (torch.cos(omega * x_high) + x_high * omega
                * torch.sin(omega * x_high)
                - torch.cos(omega * x_low) - x_low * omega
                * torch.sin(omega * x_low)) / omega**2

    @staticmethod
    def x2_cos_moment(omega, x_low, x_high):
        """
        :param omega: Oscillation frequency
        :param x_low: Lower integration limit
        :param x_high Upper integration limit
        :return: Second moment [x^2 cos(omega x)]
        """
        return ((x_high**2 * omega**2 - 2)
                * torch.sin(omega * x_high) + 2 * omega * x_high
                * torch.cos(omega * x_high)
                - (x_low**2 * omega**2 - 2)
                * torch.sin(omega * x_low) - 2 * omega * x_low
                * torch.cos(omega * x_low)) / omega**3.0


if __name__ == "__main__":
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    device = "cpu"

    track = Track("./track.npy", device=device)
    wavefnt = Wavefront(1.7526625849289021, 3.77e6,
                        [-0.01, 0.01, -0.01, 0.01],
                        [50, 50], device=device)
    slvr = EdgeRadSolver(wavefnt, track)
    slvr.auto_res()
    slvr.solve()

    wavefnt.plot_intensity()
    print(torch.sum(wavefnt.field))
    plt.show()

