import numpy as np
import matplotlib.pyplot as plt
from Track import Track
from Wavefront import Wavefront
import torch
import time
from scipy.interpolate import CubicSpline as interp1d

# Todo Change everything so that polarisation axis is the first.
#  Also change things so that the field is saved as a 2d array rather than 1

c_light = 0.29979245


class EdgeRadSolver(torch.nn.Module):
    """
    Class which solves the Liénard–Wiechert field at a wavefront for a given
    particle trajectory.
    """

    def __init__(self, wavefront, track, device=None):
        """
        :param wavefront: Instance of Wavefront class
        :param track: Instance of Track class

        """
        super().__init__()
        self.wavefront = wavefront
        self.track = track
        self.device = device

    def set_dt(self, new_samples, n_sample_x=None, flat_power=0.5):
        """
        Sets the track samples based on large values of objective function
        obj = |grad(1/grad(g))|. Where g(t) is the phase function. Takes sample
        along x dimension at y=0.
        :param new_samples: Number of new samples along trajectory
        :param n_sample_x: Approximate number of test sample points in x
        :flat_power power to which the objective function is raised. This
        increases the number of samples in the noisy bit.
        """

        # TODO change from down sampling time step to interpolating over path

        if n_sample_x:
            sample_rate = int(self.wavefront.n_samples_xy[0] / n_sample_x)
        else:
            sample_rate = 1

        # Store initial samples to print reduction
        init_samples = self.track.time.shape[0]

        # Calculate grad(1/ grad(phi)) at sample_x points.
        start_index = self.wavefront.n_samples_xy[1] // 2
        r_obs = self.wavefront.coords[
                :, start_index::self.wavefront.n_samples_xy[1]*sample_rate,
                None] - self.track.r[:, None, :]
        r_norm = torch.linalg.norm(r_obs, dim=0)
        phase_samples = self.track.time[None, :] + r_norm / c_light
        phase_grad = torch.gradient(phase_samples,
                                    spacing=(self.track.time,), dim=1)[0]
        grad_inv_grad = torch.gradient(1. / phase_grad,
                                    spacing=(self.track.time,), dim=1)[0]
        fig, ax = plt.subplots()
        ax.plot(self.track.time, 1/phase_grad.T, color="blue")

        # New samples are then evenly spaced over the cumulative distribution
        objective, _ = torch.max(torch.abs(grad_inv_grad), dim=0)
        cumulative_obj = torch.cumsum(objective**flat_power, dim=0)
        cumulative_obj = cumulative_obj / cumulative_obj[-1]

        # Now update all the samples

        track_samples = torch.linspace(0, 1, new_samples)
        self.track.time = torch.tensor(interp1d(cumulative_obj, self.track.time,
                                                )(track_samples))
        r1 = torch.tensor(interp1d(cumulative_obj, self.track.r[0],
                                                )(track_samples))
        r2 = torch.tensor(interp1d(cumulative_obj, self.track.r[1],
                                                )(track_samples))
        r3 = torch.tensor(interp1d(cumulative_obj, self.track.r[2],
                                                )(track_samples))
        self.track.r = torch.stack((r1, r2, r3))
        beta1 = torch.tensor(interp1d(cumulative_obj, self.track.beta[0],
                                                )(track_samples))
        beta2 = torch.tensor(interp1d(cumulative_obj, self.track.beta[1],
                                                )(track_samples))
        beta3 = torch.tensor(interp1d(cumulative_obj, self.track.beta[2],
                                                )(track_samples))
        self.track.beta = torch.stack((beta1, beta2, beta3))

        #track_samples = torch.linspace(0, 1, new_samples)
        #self.track.time = CubicInterp(cumulative_obj,
        #                              self.track.time)(track_samples)
        #self.track.r = CubicInterp(cumulative_obj.repeat(3, 1),
        #                           self.track.r)(track_samples.repeat(3, 1))
        #self.track.beta = CubicInterp(cumulative_obj.repeat(3, 1),
        #                            self.track.beta)(track_samples.repeat(3, 1))

        start_index = self.wavefront.n_samples_xy[1] // 2
        r_obs = self.wavefront.coords[
                :, start_index::self.wavefront.n_samples_xy[1]*sample_rate,
                None] - self.track.r[:, None, :]
        r_norm = torch.linalg.norm(r_obs, dim=0)
        phase_samples = self.track.time[None, :] + r_norm / c_light
        phase_grad = torch.gradient(phase_samples,
                                    spacing=(self.track.time,), dim=1)[0]
        grad_inv_grad = torch.gradient(1. / phase_grad,
                                    spacing=(self.track.time,), dim=1)[0]

        ax.plot(self.track.time, 1/phase_grad.T, color="red")
        plt.show()

    def solve(self, blocks=1):
        """
        Main function to solve the radiation field at the wavefront.
        Interaction limits must be within time array of track
        :param blocks: Number of blocks to split calculation. Increasing this
         will reduce memory but slow calculation
        """

        # Check array divides evenly into blocks
        if self.wavefront.coords.shape[1] % blocks != 0:
            raise Exception("Observation array does not divide evenly into "
                            f"blocks. {self.wavefront.coords.shape[1]} "
                            f"observation points and {blocks} blocks.")

        # Loop blocks and perform calculation
        block_size = int(self.wavefront.coords.shape[1] / blocks)
        for i in range(blocks):

            # start and end index of block
            bi = block_size * i
            bf = block_size * (i + 1)

            # Calculate observation points
            r_obs = self.wavefront.coords[:, bi:bf, None] \
                    - self.track.r[:, None, :]
            r_norm = torch.linalg.norm(r_obs, dim=0, keepdim=True)
            n_dir = r_obs[:2] / r_norm

            # Calculate the phase function and gradient
            phase = self.track.time + r_norm / c_light
            phase = phase - phase[:, :, None, 0]
            phase_grad = torch.unsqueeze(
                torch.gradient(torch.squeeze(phase), spacing=(self.track.time,),
                               dim=1)[0], dim=0)

            # Now calculate integrand samples
            int1 = (self.track.beta[:2, None, :] - n_dir) \
                   / (r_norm * phase_grad)
            int2 = c_light * n_dir / (self.wavefront.omega * r_norm**2.0
                                      * phase_grad)


            """"
            print(int1.shape, int2.shape)
            fig, ax = plt.subplots()
            ax.plot(self.track.time, int1[0].T)
            plt.show()
            input()
            """




            real_part = (self.filon_cos(phase, int1, self.wavefront.omega)
                         + self.filon_sin(phase, int2, self.wavefront.omega))
            imag_part = (self.filon_sin(phase, int1, self.wavefront.omega)
                         - self.filon_cos(phase, int2, self.wavefront.omega))
            self.wavefront.field[:, bi:bf] = (real_part + 1j * imag_part)

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


class CubicInterp:
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
    # Todo Make the calulcation single precision
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    device = "cpu"

    track = Track(device=device)
    track.load_file("./../EdgeRadSolver1/track_test.npy")
    print(track.r.shape)

    wavefnt = Wavefront(1.7526625849289021, 3.77e6,
                        [-0.01, 0.01, -0.01, 0.01],
                        [1000, 1000], device=device)

    slvr = EdgeRadSolver(wavefnt, track)


    slvr.set_dt(50, flat_power=0.5)
    slvr.solve()
    wavefnt.plot_intensity()

    plt.show()


