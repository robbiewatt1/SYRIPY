import matplotlib.pyplot as plt
import torch
from .Wavefront import Wavefront
from .Tracking import Track
from .Interpolation import CubicInterp
from typing import Optional, Tuple
import copy


class FieldSolver(torch.nn.Module):
    """
    Class which solves the Liénard–Wiechert field at a wavefront for a given
    particle trajectory. Field is calculated assuming 1 nc of charge and SI
    units apart from time in ns.
    """

    def __init__(self, wavefront: Wavefront, track: Track,
                 blocks: int = 1) -> None:
        """
        :param wavefront: Instance of Wavefront class
        :param track: Instance of Track class
        :param blocks: Number of blocks to split calculation. Increasing this
         will reduce memory but slow calculation.
        """
        super().__init__()
        self.wavefront = wavefront
        self.track = track
        self.blocks = blocks
        self.c_light = 0.299792458

        # Check that wavefront / track device are both the same
        if track.device != wavefront.device:
            raise Exception("Track and wavefront are on different devices!")
        self.device = wavefront.device

        # Check array divides evenly into blocks
        if self.wavefront.coords.shape[1] % self.blocks != 0:
            raise Exception("Observation array does not divide evenly into "
                            f"blocks. {self.wavefront.coords.shape[1]} "
                            f"observation points and {self.blocks} blocks.")

    @torch.jit.export
    def switch_device(self, device: torch.device) -> "FieldSolver":
        """
        Changes the device that the class data is stored on.
        :param device: Device to switch to.
        """
        self.device = device
        self.track.switch_device(device)
        self.wavefront.switch_device(device)
        return self

    @torch.jit.ignore
    def set_track(self, new_samples: int, t_start: Optional[float] = None,
                  t_end: Optional[float] = None,
                  n_sample: Optional[float] = None,
                  flat_power: float = 0.25, mode: str = "nn") -> None:
        """
        Sets the track samples based on large values of objective function
        obj = |grad(1/grad(g))|. Where g(t) is the phase function. Takes sample
        along x dimension at y=0. Also calculates the radius of curvature of the
        wavefront.
        :param new_samples: Number of new samples along trajectory.
        :param t_start: Start time for integration. If t_start=None start
         time of original track will be used.
        :param t_end: End time for integration. If t_end=None end time of
         original track will be used.
        :param n_sample: Approximate number of test sample points along x
         dimension.
        :param flat_power: Power to which the objective function is raised. This
         increases the number of samples in the noisy bit.
        :param mode: Interpolation mode. Can be cubic or nn (nearest neighbor).
        """

        if new_samples % 2 == 0:
            print(f"Warning: Filon integrator is a Simpson's based method"
                  f" requiring an odd number of time steps for high accuracy."
                  f" Increasing new_samples to {new_samples + 1}")
            new_samples += 1

        if mode.lower() not in ["cubic", "nn"]:
            raise Exception(f"Unknown interpolation mode {mode}. Please use "
                            f"either \"cubic\" or \"nn\".")

        # Reset the wavefront
        self.wavefront.reset()

        if n_sample is not None:
            sample_rate = int(self.wavefront.n_samples_xy[0] / n_sample)
        else:
            sample_rate = 1

        # Set time start / end and find the closest indexes
        if t_start is None:
            t_start = self.track.time[0]
        if t_end is None:
            t_end = self.track.time[-1]

        t_0 = torch.argmin(torch.abs(self.track.time - t_start))
        t_1 = torch.argmin(torch.abs(self.track.time - t_end)) + 1

        # Calculate grad(1/ grad(phi)) at sample_x points.
        start_index = self.wavefront.n_samples_xy[1] // 2
        r_obs = self.wavefront.coords[
                :, start_index::self.wavefront.n_samples_xy[1]*sample_rate,
                None] - self.track.r[:, None, t_0:t_1]

        rb_xy = torch.sum(r_obs[:2] * self.track.beta[:2, None, t_0:t_1], dim=0)
        r2_xy = torch.sum(r_obs[:2] * r_obs[:2], dim=0)
        b2_xy = torch.sum(self.track.beta[:2, None, t_0:t_1]
                          * self.track.beta[:2, None, t_0:t_1], dim=0)
        phase_grad = ((self.track.gamma**-2 + b2_xy) / 2.
                      - (2. * r_obs[2] * rb_xy - r2_xy
                         * self.track.beta[2, None, t_0:t_1])
                      / (2 * r_obs[2] * r_obs[2] + r2_xy))

        grad_inv_grad = torch.gradient(1. / phase_grad, spacing=(
            self.track.time[t_0:t_1],), edge_order=1, dim=1)[0]

        # New samples are then evenly spaced over the cumulative distribution
        objective, _ = torch.max(torch.abs(grad_inv_grad), dim=0)
        cumulative_obj = torch.cumsum(objective**flat_power, dim=0)
        cumulative_obj = (cumulative_obj - cumulative_obj[0]) \
                         / (cumulative_obj[-1] - cumulative_obj[0])

        # Use the objective function to calculate the radius of curvature
        peak_index = t_0 + int(torch.sum(objective * torch.arange(
            objective.shape[0], device=self.device)) / torch.sum(objective))
        wf_centre = [
            (self.wavefront.wf_bounds[1] + self.wavefront.wf_bounds[0]) / 2,
            (self.wavefront.wf_bounds[3] + self.wavefront.wf_bounds[2]) / 2,
            self.wavefront.z]
        self.wavefront.curv_r = (
                (self.track.r[0, peak_index] - wf_centre[0])**2.
              + (self.track.r[1, peak_index] - wf_centre[1])**2.
              + (self.track.r[2, peak_index] - wf_centre[2])**2.)**0.5

        # Now update all the samples
        track_samples = torch.linspace(0., 1, new_samples, device=self.device)
        if mode == "cubic":  # Use cubic interpolation
            self.track.time = CubicInterp(
                cumulative_obj, self.track.time[t_0:t_1])(track_samples)
            self.track.r = CubicInterp(
                cumulative_obj.repeat(3, 1), self.track.r[:, t_0:t_1]
                )(track_samples.repeat(3, 1))
            self.track.beta = CubicInterp(
                cumulative_obj.repeat(3, 1), self.track.beta[:, t_0:t_1]
                )(track_samples.repeat(3, 1))

            # If bunch track exists then also down sample
            if self.track.bunch_r.shape[0] != 0:
                n = self.track.bunch_r.shape[:2]
                self.track.bunch_r = CubicInterp(
                  cumulative_obj.repeat(*n, 1), self.track.bunch_r[..., t_0:t_1]
                  )(track_samples.repeat(*n, 1))
                self.track.bunch_beta = CubicInterp(
                    cumulative_obj.repeat(*n, 1),
                    self.track.bunch_beta[..., t_0:t_1]
                    )(track_samples.repeat(*n, 1))

        else:  # Use nearest neighbour interpolation
            sample_index = torch.unique(torch.abs(
                cumulative_obj[:, None] - track_samples[None]).argmin(dim=0))

            # We need an odd number of samples to avoid strange things happening
            if len(sample_index) % 2 == 0:
                sample_index = sample_index[:-1]

            self.track.time = self.track.time[t_0:t_1][sample_index]
            self.track.r = self.track.r[:, t_0:t_1][:, sample_index]
            self.track.beta = self.track.beta[:, t_0:t_1][:, sample_index]

            # If bunch track exists then also down sample
            if self.track.bunch_r.shape[0] != 0:
                self.track.bunch_r = self.track.bunch_r[..., t_0:t_1][
                    ..., sample_index]
                self.track.bunch_beta = self.track.bunch_beta[..., t_0:t_1][
                    ..., sample_index]

        # Set time to be 0 at start
        self.track.time = self.track.time - self.track.time[0]

    @torch.jit.export
    def solve_field(self, bunch_index: int = -1, solve_ends: bool = True
                    ) -> Wavefront:
        """
        Main callable function to solve the field for either a single particle
        or a sample from a bunch.
        :param bunch_index: Index of particle to solve. If we are just solving a
         single central trajectory then bunch_index=-1 which is the default.
        :param solve_ends: If true then we use a first order asymptotic
         expansion to solve the ends of the integral.
        :return: The wavefront with calculated field array.
        """

        if bunch_index == -1:
            if self.track.r.shape[0] == 0:
                raise Exception(
                    "Cannot calculate particle field because track hasn't "
                    "been simulated yet. Please simulate track with "
                    "track.sim_single_c() or track.sim_single() before trying "
                    "to solve field.")
            r = self.track.r
            beta = self.track.beta
            gamma = self.track.gamma
        else:
            if self.track.bunch_r.shape[0] == 0:
                raise Exception(
                    "Cannot calculate bunch sample field because bunch tracks "
                    "haven't been simulated yet. Please simulate bunch track "
                    "with  track.sim_bunch_c() or track.sim_bunch() before "
                    "trying to solve field.")
            r = self.track.bunch_r[bunch_index]
            beta = self.track.bunch_beta[bunch_index]
            gamma = self.track.bunch_gamma[bunch_index]
        return self._solve_field(self.track.time, r, beta, gamma, solve_ends,
                                 solve_ends)

    @torch.jit.export
    def solve_field_vmap(self, bunch_index: torch.Tensor,
                         solve_ends: bool = True) -> Wavefront:
        """
        Main callable function to solve the field if we are using torch's vmap
        method to small simulations in batch mode.
        :param bunch_index: Batch tensor of indices to be solved.
        :param solve_ends: If true then we use a first order asymptotic
         expansion to solve the ends of the integral.
        :return: The wavefront with calculated field array.
        """
        r = self.track.bunch_r[bunch_index[None]][0]
        beta = self.track.bunch_beta[bunch_index[None]][0]
        gamma = self.track.bunch_gamma[bunch_index[None]][0]
        return self._solve_field(self.track.time, r, beta, gamma, solve_ends,
                                 solve_ends)

    def forward(self, time: torch.Tensor,  r: torch.Tensor, beta: torch.Tensor,
                gamma: torch.Tensor, solve_ends: bool = True) -> Wavefront:
        """
        Forward function is a wrapper for the base solver function _solve_field.
        Doesn't depend on class track variables, so can be useful for compiling
        :param time: Time samples along track.
        :param r: Position samples along track.
        :param beta: Velocity samples along track.
        :param gamma: Particle lorentz factor.
        :param solve_ends: If true then we use a first order asymptotic
         expansion to solve the ends of the integral to infinity.
        """
        return self._solve_field(time, r, beta, gamma, solve_ends, solve_ends)

    @torch.jit.export
    def solve_field_split(self, bunch_index: int = -1, split_time: float = 0,
                          plot_track: bool = False):
        """
        Splits the track into two parts at time index closest to split_time and
        calculates a separate field for both parts of the track.
        :param bunch_index: Batch tensor of indices to be solved.
         expansion to solve the ends of the integral.
        :param split_time: Time when the track is split.
        :param plot_track: Plots the split track to check time index is in the
         right place.
        :return: (wavefront_1, wavefront_2) with updated field array
        """
        if bunch_index == -1:
            if self.track.r.shape[0] == 0:
                raise Exception(
                    "Cannot calculate particle field because track hasn't "
                    "been simulated yet. Please simulate track with "
                    "track.sim_single_c() or track.sim_single() before trying "
                    "to solve field.")
            r = self.track.r
            beta = self.track.beta
            gamma = self.track.gamma
        else:
            if self.track.bunch_r.shape[0] == 0:
                raise Exception(
                    "Cannot calculate bunch sample field because bunch tracks "
                    "haven't been simulated yet. Please simulate bunch track "
                    "with  track.sim_bunch_c() or track.sim_bunch() before "
                    "trying to solve field.")
            r = self.track.bunch_r[bunch_index]
            beta = self.track.bunch_beta[bunch_index]
            gamma = self.track.bunch_gamma[bunch_index]

        if split_time < self.track.time[0] or split_time > self.track.time[-1]:
            raise Exception(
                f"Split time: {split_time}, is outside the range of the track. "
                f"i.e. min = {self.track.time[0]}, max = {self.track.time[-1]}."
            )

        split_index = torch.argmin(torch.abs(self.track.time - split_time))
        if split_index % 2 == 0:
            split_index += 1

        wavefront_1 = copy.deepcopy(self._solve_field(
            self.track.time[:split_index], r[:, :split_index],
            beta[:, :split_index], gamma, True, False))
        wavefront_2 = self._solve_field(
            self.track.time[split_index-1:], r[:, split_index-1:],
            beta[:, split_index-1:], gamma, False, True)

        if plot_track:
            fig, ax = plt.subplots()
            ax.plot(r[2, :split_index].cpu().detach(),
                    r[0, :split_index].cpu(), color="red")
            ax.plot(r[2, split_index-1:].cpu().detach(),
                    r[0, split_index-1:].cpu().detach(), color="blue")
            return wavefront_1, wavefront_2, (fig, ax)
        else:
            return wavefront_1, wavefront_2

    @torch.jit.export
    def _solve_field(self, time: torch.Tensor, r: torch.Tensor,
                     beta: torch.Tensor, gamma: torch.Tensor,
                     solve_ends_l: bool = True,
                     solve_ends_r: bool = True) -> Wavefront:
        """
        Function which solves the integral. However, this shouldn't be called
        directly as it requires track information as an argument.
        :param time: Time samples along track.
        :param r: Position samples along track.
        :param beta: Velocity samples along track.
        :param gamma: Particle lorentz factor.
        :param solve_ends_l: If true then we use a first order asymptotic
         expansion to solve the left ends of the integral to infinity.
        :param solve_ends_r: If true then we use a first order asymptotic
         expansion to solve the right ends of the integral to infinity.
        """

        # Reset the wavefront
        self.wavefront.reset()

        # Loop blocks and perform calculation
        block_size = self.wavefront.coords.shape[1] // self.blocks
        for i in range(self.blocks):

            # start and end index of block
            bi = block_size * i
            bf = block_size * (i + 1)

            # Calculate observation points
            r_obs = self.wavefront.coords[:, bi:bf, None] - r[:, None, :]

            # Calculate the gradient of the phase
            rb_xy = torch.sum(r_obs[:2] * beta[:2, None, :], dim=0)
            r2_xy = torch.sum(r_obs[:2] * r_obs[:2], dim=0)
            b2_xy = torch.sum(beta[:2] * beta[:2], dim=0)
            phase_grad = ((gamma**-2 + b2_xy) / 2. - (
                    2. * r_obs[2] * rb_xy - r2_xy * beta[2, None, :])
                          / (2 * r_obs[2] * r_obs[2] + r2_xy))[None]
            phase, delta_phase = self.cumulative_trapz(time, phase_grad)

            # Now calculate integrand samples
            r_norm = (2. * r_obs[2]**2. + r2_xy) / (2. * r_obs[2])
            n_dir = r_obs[:2] / r_norm
            int1 = (beta[:2, None, :] - n_dir) / r_norm
            int2 = - self.c_light * n_dir / (self.wavefront.omega * r_norm**2.0)

            # Solve main part of integral
            imag1, real1 = self.filon_method(
                phase, delta_phase, int1 / phase_grad, self.wavefront.omega)
            real2, imag2 = self.filon_method(
                phase, delta_phase, int2 / phase_grad, self.wavefront.omega)

            field = (real1 - real2 + 1j * (imag1 + imag2))

            # Solve end points to inf
            if solve_ends_l:
                f_l = int1[:, :, 0] + 1j * int2[:, :, 0]
                field -= (torch.exp(1j * self.wavefront.omega * phase[:, :, 0])
                          * f_l / phase_grad[:, :, 0]) * 1j \
                         / self.wavefront.omega
            if solve_ends_r:
                f_r = int1[:, :, -1] + 1j * int2[:, :, -1]
                field += (torch.exp(1j * self.wavefront.omega * phase[:, :, -1])
                         * f_r / phase_grad[:, :, -1]) * 1j \
                         / self.wavefront.omega

            # Add in initial phase part
            field = field * torch.exp(1j * r2_xy[:, 0] / (2. * r_obs[2, :, 0])
                                      * self.wavefront.omega / self.c_light
                                      ) * self.wavefront.omega

            # First method doesn't work with vmap so need this
            if self.blocks > 1:
                self.wavefront.field[:, bi:bf] = field
            else:
                self.wavefront.field = field
        return self.wavefront

    @staticmethod
    def filon_method(x_samples: torch.Tensor, delta_x: torch.Tensor,
                     f_samples: torch.Tensor, omega: float
                     ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Filon based method for integrating function multiplied by a rapidly
        oscillating sine wave. I = int[f(x) sin(omega x)], omega >> 1. Uses a
        quadratic approximation for f(x), allowing I to be solved analytically.
        :param x_samples: Sample points of integration.
        :param delta_x: Difference between samples.
        :param f_samples: Samples of non-oscillating function.
        :param omega: Oscillation frequency.
        :return: Integration result.
        """
        sin_x1 = torch.sin(omega * x_samples[..., 1:-1:2])
        cos_x1 = torch.cos(omega * x_samples[..., 1:-1:2])
        sin_x1x0 = torch.sin(omega * delta_x[..., ::2])
        cos_x1x0 = torch.cos(omega * delta_x[..., ::2])
        sin_x2x1 = torch.sin(omega * delta_x[..., 1::2])
        cos_x2x1 = torch.cos(omega * delta_x[..., 1::2])
        x1x0 = omega * delta_x[..., ::2]
        x2x1 = omega * delta_x[..., 1::2]
        x2x0 = omega * (delta_x[..., 1::2] + delta_x[..., :-1:2])

        w0_sin = (sin_x1 * ((x2x0 * x1x0 - 2.) * sin_x1x0 + (x1x0 + x2x0)
                            * cos_x1x0 - 2. * sin_x2x1 + x2x1 * cos_x2x1)
                  + cos_x1 * ((x2x0 * x1x0 - 2.) * cos_x1x0 - (x1x0 + x2x0)
                              * sin_x1x0 + 2. * cos_x2x1 + x2x1 * sin_x2x1)
                  ) / (omega * x2x0 * x1x0)
        w1_sin = (sin_x1 * (x2x0 * (cos_x2x1 + cos_x1x0)
                            - 2. * (sin_x2x1 + sin_x1x0))
                  + cos_x1 * (x2x0 * (sin_x2x1 - sin_x1x0)
                              + 2. * (cos_x2x1 - cos_x1x0))
                  ) / (omega * x1x0 * -x2x1)
        w2_sin = (sin_x1 * ((2. - x2x0 * x2x1) * sin_x2x1 - (x2x1 + x2x0)
                            * cos_x2x1 + 2. * sin_x1x0 - x1x0 * cos_x1x0)
                  + cos_x1 * ((x2x0 * x2x1 - 2.) * cos_x2x1 - (x2x1 + x2x0)
                              * sin_x2x1 + 2. * cos_x1x0 + x1x0 * sin_x1x0)
                  ) / (omega * -x2x0 * x2x1)
        w0_cos = (sin_x1 * (cos_x1x0 * (2. - x2x0 * x1x0) + sin_x1x0
                            * (x1x0 + x2x0) - 2. * cos_x2x1 - sin_x2x1 * x2x1)
                  + cos_x1 * (sin_x1x0 * (x2x0 * x1x0 - 2.) + cos_x1x0
                              * (x1x0 + x2x0) - 2. * sin_x2x1 + cos_x2x1 * x2x1)
                  ) / (omega * x1x0 * x2x0)
        w1_cos = (sin_x1 * (x2x0 * (sin_x1x0 - sin_x2x1)
                            + 2 * (-cos_x2x1 + cos_x1x0))
                  + cos_x1 * (x2x0 * (cos_x1x0 + cos_x2x1)
                              - 2 * (sin_x2x1 + sin_x1x0))
                  ) / (omega * -x1x0 * x2x1)
        w2_cos = (sin_x1 * ((x2x0 * x2x1 - 2.) * cos_x2x1 - (x2x1 + x2x0)
                            * sin_x2x1 + 2. * cos_x1x0 + x1x0 * sin_x1x0)
                  + cos_x1 * ((x2x0 * x2x1 - 2.) * sin_x2x1 + (x2x1 + x2x0)
                              * cos_x2x1 - 2. * sin_x1x0 + x1x0 * cos_x1x0)
                  ) / (omega * x2x0 * x2x1)

        return (torch.sum(w0_sin * f_samples[..., :-2:2]
                          + w1_sin * f_samples[..., 1:-1:2]
                          + w2_sin * f_samples[..., 2::2], dim=-1),
                torch.sum(w0_cos * f_samples[..., :-2:2]
                          + w1_cos * f_samples[..., 1:-1:2]
                          + w2_cos * f_samples[..., 2::2], dim=-1))

    @staticmethod
    def cumulative_trapz(x: torch.Tensor, y: torch.Tensor):
        """
        Function to calculate the cumulative integral of y w.r.t x. Uses a first
        order trapz method.
        :param x: Input axis samples.
        :param y: Integrand samples at location x.
        :return: Cumulative integral of y starting at 0.
        """
        delta_x = torch.diff(x, dim=-1)
        delta_y = delta_x * (y[..., 1:] + y[..., :-1]) / 2.
        result = torch.zeros_like(y)
        result[..., 1:] = torch.cumsum(delta_y, dim=-1)
        return result, delta_y
