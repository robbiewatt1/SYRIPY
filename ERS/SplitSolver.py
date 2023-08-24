import matplotlib.pyplot as plt
import torch
from .Wavefront import Wavefront
from .Tracking import Track
from .Interpolation import CubicInterp

from typing import Optional, Tuple
import copy


class SplitSolver(torch.nn.Module):
    """
    Class which solves the Liénard–Wiechert field at a wavefront for a given
    particle trajectory. This does the same calculation as FieldSolver, however
    it splits the track at a given time and returns two wavefronts
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
        self.split_index = None
        self.wavefront_curve = None
        self.source_location = None

        # Check that wavefront / track device are both the same
        if track.device != wavefront.device:
            raise Exception("Track and wavefront are on different devices!")
        self.device = wavefront.device

        # Check array divides evenly into blocks
        if self.wavefront.coords.shape[1] % self.blocks != 0:
            raise Exception("Observation array does not divide evenly into "
                            f"blocks. {self.wavefront.coords.shape[1]} "
                            f"observation points and {self.blocks} blocks.")

        self.wf_centre = [
            (self.wavefront.wf_bounds[1] + self.wavefront.wf_bounds[0]) / 2,
            (self.wavefront.wf_bounds[3] + self.wavefront.wf_bounds[2]) / 2,
            self.wavefront.z]

    @torch.jit.export
    def switch_device(self, device: torch.device) -> "SplitSolver":
        """
        Changes the device that the class data is stored on.
        :param device: Device to switch to.
        """
        self.device = device
        self.track.switch_device(device)
        self.wavefront.switch_device(device)
        return self

    @torch.jit.ignore
    def set_track(self, t_split: float, new_samples: int,
                  t_start: Optional[float] = None,
                  t_end: Optional[float] = None,
                  n_sample: Optional[float] = None, flat_power: float = 0.25,
                  mode: str = "nn", plot_track: bool = False) -> None:
        """
        Calculates the split index of the track and sets the track samples
        based on large values of objective function obj = |grad(1/grad(g))|.
        Where g(t) is the phase function. Takes sample along x dimension at y=0.
        Also calculates the radius of curvature of the  wavefront.
        :param t_split: Time at which the track is split
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
        :param plot_track: Plots the split track to check time index is in the
         right place.
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
        split_index = torch.argmin(torch.abs(self.track.time[t_0:t_1]
                                             - t_split))

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

        objective_1, _ = torch.max(torch.abs(
            grad_inv_grad[:, :split_index]), dim=0)
        peak_index_1 = t_0 + int(torch.sum(objective_1 * torch.arange(
            objective_1.shape[0], device=self.device)) / torch.sum(objective_1))
        curv_1 = ((self.track.r[0, peak_index_1] - self.wf_centre[0])**2.
                 + (self.track.r[1, peak_index_1] - self.wf_centre[1])**2.
                 + (self.track.r[2, peak_index_1] - self.wf_centre[2])**2.)**0.5

        objective_2, _ = torch.max(torch.abs(
            grad_inv_grad[:, split_index:]), dim=0)
        peak_index_2 = t_0 + split_index + int(torch.sum(
            objective_2 * torch.arange(objective_2.shape[0], device=self.device)
            ) / torch.sum(objective_2))
        curv_2 = ((self.track.r[0, peak_index_2] - self.wf_centre[0])**2.
                 + (self.track.r[1, peak_index_2] - self.wf_centre[1])**2.
                 + (self.track.r[2, peak_index_2] - self.wf_centre[2])**2.)**0.5
        self.wavefront_curve = [curv_1, curv_2]
        self.source_location = [self.track.r[:, peak_index_1],
                                self.track.r[:, peak_index_2]]
        print(self.wavefront_curve,
              self.source_location)

        # New samples are then evenly spaced over the cumulative distribution
        objective, _ = torch.max(torch.abs(grad_inv_grad), dim=0)
        cumulative_obj = torch.cumsum(objective**flat_power, dim=0)
        cumulative_obj = (cumulative_obj - cumulative_obj[0]) \
                         / (cumulative_obj[-1] - cumulative_obj[0])

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

        # Update split index for new track samples and set time to start at 0
        self.split_index = torch.argmin(torch.abs(self.track.time - t_split)
                                        ) // 2
        self.track.time = self.track.time - self.track.time[0]

        # Here we plot the split track if required
        if plot_track:
            fig, ax = plt.subplots()
            ax.plot(self.track.r[2, self.split_index*2:].cpu().detach(),
                    self.track.r[0, self.split_index*2:].cpu().detach(),
                    color="red")
            ax.plot(self.track.r[2, :self.split_index*2].cpu().detach(),
                    self.track.r[0, :self.split_index*2].cpu().detach(),
                    color="blue")

    @torch.jit.export
    def solve_field(self, bunch_index: int = -1, solve_ends: bool = True
                    ) -> Tuple[Wavefront, Wavefront]:
        """
        Splits the track into two parts at time index closest to split_time and
        calculates a separate field for both parts of the track.
        :param bunch_index: Batch tensor of indices to be solved.
         expansion to solve the ends of the integral.
        :param solve_ends: If true then ends of the
        :return: (wavefront_1, wavefront_2) with updated field array
        """

        if self.split_index is None:
            raise Exception("Set_track function must be called before field"
                            " can be calculated")

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
    def _solve_field(self, time: torch.Tensor, r: torch.Tensor,
                     beta: torch.Tensor, gamma: torch.Tensor,
                     solve_ends_l: bool = True, solve_ends_r: bool = True
                     ) -> Tuple[Wavefront, Wavefront]:
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
        wavefront1 = copy.deepcopy(self.wavefront)
        wavefront2 = copy.deepcopy(self.wavefront)
        wavefront1.reset()
        wavefront2.reset()
        wavefront1.curv_r = self.wavefront_curve[0]
        wavefront2.curv_r = self.wavefront_curve[1]
        wavefront1.source_location = self.source_location[0]
        wavefront2.source_location = self.source_location[1]

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
            imag1_wf1, imag1_wf2, real1_wf1, real1_wf2 = self.filon_method(
                phase, delta_phase, int1 / phase_grad, self.wavefront.omega,
                self.split_index)
            imag2_wf1, imag2_wf2, real2_wf1, real2_wf2 = self.filon_method(
                phase, delta_phase, int2 / phase_grad, self.wavefront.omega,
                self.split_index)

            field1 = (real1_wf1 - real2_wf1 + 1j * (imag1_wf1 + imag2_wf1))
            field2 = (real1_wf2 - real2_wf2 + 1j * (imag1_wf2 + imag2_wf2))

            # Solve end points to inf
            if solve_ends_l:
                f_l = int1[:, :, 0] + 1j * int2[:, :, 0]
                f_r = int1[:, :, self.split_index*2] + 1j\
                      * int2[:, :, self.split_index*2]
                field1 -= (torch.exp(1j * self.wavefront.omega * phase[:, :, 0])
                           * f_l / phase_grad[:, :, 0]) * 1j \
                          / self.wavefront.omega
                field1 += (torch.exp(1j * self.wavefront.omega
                                     * phase[:, :, self.split_index*2]) * f_r
                           / phase_grad[:, :, self.split_index*2]) * 1j \
                          / self.wavefront.omega
            if solve_ends_r:
                f_l = int1[:, :, self.split_index*2] + 1j\
                      * int2[:, :, self.split_index*2]
                f_r = int1[:, :, -1] + 1j * int2[:, :, -1]
                field2 -= (torch.exp(1j * self.wavefront.omega
                                     * phase[:, :, self.split_index*2])
                           * f_l / phase_grad[:, :, self.split_index*2]) * 1j \
                          / self.wavefront.omega
                field2 += (torch.exp(1j * self.wavefront.omega
                                     * phase[:, :, -1])
                           * f_r / phase_grad[:, :, -1]) * 1j \
                          / self.wavefront.omega

            # Add in initial phase part
            field1 = field1 * torch.exp(1j * r2_xy[:, 0] / (2. * r_obs[2, :, 0])
                                        * self.wavefront.omega / self.c_light
                                        ) * self.wavefront.omega
            field2 = field2 * torch.exp(1j * r2_xy[:, 0] / (2. * r_obs[2, :, 0])
                                        * self.wavefront.omega / self.c_light
                                        ) * self.wavefront.omega

            # First method doesn't work with vmap so need this
            if self.blocks > 1:
                wavefront1.field[:, bi:bf] = field1
                wavefront2.field[:, bi:bf] = field2
            else:
                wavefront1.field = field1
                wavefront2.field = field2
        return wavefront1, wavefront2

    @staticmethod
    def filon_method(x_samples: torch.Tensor, delta_x: torch.Tensor,
                     f_samples: torch.Tensor, omega: float, split_index: int,
                     ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor,
                                torch.Tensor]:
        """
        Filon based method for integrating function multiplied by a rapidly
        oscillating sine wave. I = int[f(x) sin(omega x)], omega >> 1. Uses a
        quadratic approximation for f(x), allowing I to be solved analytically.
        :param x_samples: Sample points of integration.
        :param delta_x: Difference between samples.
        :param f_samples: Samples of non-oscillating function.
        :param omega: Oscillation frequency.
        :param split_index: Index at which track is split.
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
        sin_term = w0_sin * f_samples[..., :-2:2] \
                   + w1_sin * f_samples[..., 1:-1:2] \
                   + w2_sin * f_samples[..., 2::2]
        cos_term = w0_cos * f_samples[..., :-2:2] \
                   + w1_cos * f_samples[..., 1:-1:2] \
                   + w2_cos * f_samples[..., 2::2]
        return (torch.sum(sin_term[..., :split_index], dim=-1),
                torch.sum(sin_term[..., split_index:], dim=-1),
                torch.sum(cos_term[..., :split_index], dim=-1),
                torch.sum(cos_term[..., split_index:], dim=-1))

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
