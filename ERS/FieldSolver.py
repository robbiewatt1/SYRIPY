import matplotlib.pyplot as plt
import torch


# TODO Change things so that the field is saved as a 2d array rather than 1
# TODO Add checks to make sure track has already been solved

c_light = 0.29979245


class FieldSolver(torch.nn.Module):
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

    def set_dt(self, new_samples, n_sample_x=None, flat_power=0.5,
               set_bunch=False):
        """
        Sets the track samples based on large values of objective function
        obj = |grad(1/grad(g))|. Where g(t) is the phase function. Takes sample
        along x dimension at y=0.
        :param new_samples: Number of new samples along trajectory
        :param n_sample_x: Approximate number of test sample points in x
        :param flat_power: Power to which the objective function is raised. This
        increases the number of samples in the noisy bit.
        :param set_bunch: If true then bunch track is also interpolated.
        """

        # Check that bunch trajectories have already been set if trying to
        # interpolate them
        if set_bunch and self.track.bunch_r is None:
            raise Exception("Bunch trajectories have to be set before trying to"
                            " interpolate them. Use set_bunch=False or run"
                            " track.sim_bunch_c().")

        if n_sample_x:
            sample_rate = int(self.wavefront.n_samples_xy[0] / n_sample_x)
        else:
            sample_rate = 1

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

        # New samples are then evenly spaced over the cumulative distribution
        objective, _ = torch.max(torch.abs(grad_inv_grad), dim=0)
        cumulative_obj = torch.cumsum(objective**flat_power, dim=0)
        cumulative_obj = cumulative_obj / cumulative_obj[-1]

        # Now update all the samples
        track_samples = torch.linspace(0, 1, new_samples, device=self.device)
        self.track.time = CubicInterp(cumulative_obj,
                                      self.track.time)(track_samples)
        self.track.r = CubicInterp(cumulative_obj.repeat(3, 1),
                                   self.track.r)(track_samples.repeat(3, 1))
        self.track.beta = CubicInterp(cumulative_obj.repeat(3, 1),
                                    self.track.beta)(track_samples.repeat(3, 1))

        if set_bunch:
            n = self.track.bunch_r.shape[:2]
            self.track.bunch_r = CubicInterp(cumulative_obj.repeat(*n, 1),
                                self.track.bunch_r)(track_samples.repeat(*n, 1))
            self.track.bunch_beta = CubicInterp(cumulative_obj.repeat(*n, 1),
                             self.track.bunch_beta)(track_samples.repeat(*n, 1))

    def solve(self, blocks=1, solve_ends=True, return_new=False,
              bunch_index=None):
        """
        Main function to solve the radiation field at the wavefront.
        Interaction limits must be within time array of track
        :param blocks: Number of blocks to split calculation. Increasing this
         will reduce memory but slow calculation
        :param solve_ends: If true the integration is extended to +/- inf using
         an asymptotic expansion.
        :param return_new: If true a new wavefront is return, else the class
         wavefront is returned.
        :param bunch_index: Index of bunch track. If none then central track is
         used.
        """

        # Check array divides evenly into blocks
        if self.wavefront.coords.shape[1] % blocks != 0:
            raise Exception("Observation array does not divide evenly into "
                            f"blocks. {self.wavefront.coords.shape[1]} "
                            f"observation points and {blocks} blocks.")

        # Check if we are returning new wavefront or old one
        if return_new:
            wavefront = self.wavefront.copy()
        else:
            wavefront = self.wavefront

        # Check if we are sampling from bunch track or central track
        if bunch_index is None:
            r = self.track.r
            beta = self.track.beta
        else:
            r = self.track.bunch_r[bunch_index]
            beta = self.track.bunch_beta[bunch_index]

        # Loop blocks and perform calculation
        block_size = int(wavefront.coords.shape[1] / blocks)
        for i in range(blocks):

            # start and end index of block
            bi = block_size * i
            bf = block_size * (i + 1)

            # Calculate observation points
            r_obs = wavefront.coords[:, bi:bf, None] - r[:, None, :]
            r_norm = torch.linalg.norm(r_obs, dim=0, keepdim=True)
            n_dir = r_obs[:2] / r_norm

            # Calculate the phase function and gradient
            phase = self.track.time + r_norm / c_light
            phase0 = phase[:, :, 0]
            phase = phase - phase0[..., None]
            phase_grad = torch.unsqueeze(
                torch.gradient(torch.squeeze(phase), spacing=(self.track.time,),
                               dim=1)[0], dim=0)

            # Now calculate integrand samples
            int1 = (beta[:2, None, :] - n_dir) / (r_norm * phase_grad)
            int2 = c_light * n_dir / (wavefront.omega * r_norm**2.0
                                      * phase_grad)

            # Solve main part of integral
            real_part = (self.filon_cos(phase, int1, wavefront.omega)
                         + self.filon_sin(phase, int2, wavefront.omega))
            imag_part = (self.filon_sin(phase, int1, wavefront.omega)
                         - self.filon_cos(phase, int2, wavefront.omega))

            # Solve end points to inf
            if solve_ends:
                real_part += (torch.cos(wavefront.omega * phase[..., 0])
                              * int2[..., 0]
                              - torch.cos(wavefront.omega * phase[..., -1])
                              * int2[..., -1]
                              + torch.sin(wavefront.omega * phase[..., 0])
                              * int1[..., 0]
                              - torch.sin(wavefront.omega * phase[..., -1])
                              * int1[..., -1]) / wavefront.omega
                imag_part += (torch.cos(wavefront.omega * phase[..., -1])
                              * int1[..., -1]
                              - torch.cos(wavefront.omega * phase[..., 0])
                              * int1[..., 0]
                              - torch.sin(wavefront.omega * phase[..., -1])
                              * int2[..., -1]
                              + torch.sin(wavefront.omega * phase[..., 0])
                              * int2[..., 0]) / wavefront.omega

            wavefront.field[:, bi:bf] = (real_part + 1j * imag_part) \
                                        * torch.exp(1j * wavefront.omega
                                                    * phase0)
        return wavefront

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
