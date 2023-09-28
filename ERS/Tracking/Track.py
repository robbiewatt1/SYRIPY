import numpy as np
import torch
import torch.linalg
from torch.distributions.multivariate_normal import MultivariateNormal
from torch.distributions.normal import Normal
import matplotlib.pyplot as plt
from .cTrack import cTrack
from .Magnets import FieldContainer
from typing import Optional, List, Tuple


class Track(torch.nn.Module):
    """
    Class which handles tracking of an electron through the magnetic setup. This
    can either be solved using an RK4 method or loaded from an external file.
    """

    def __init__(self, field_container: Optional[FieldContainer] = None,
                 device: Optional[torch.device] = None) -> None:
        """
        :param device: Device being used (cpu / gpu)
        """
        super().__init__()
        self.field_container = field_container
        self.device = device

        # Here we define all the track parameters. We initialise them as empty
        # tensors so torch jit doesn't get mad
        self.time = torch.tensor([])   # Proper time along particle path
        self.r = torch.tensor([])      # Particle position
        self.p = torch.tensor([])      # Particle momentum
        self.beta = torch.tensor([])   # Velocity along particle path
        self.gamma = torch.tensor([])  # Lorentz factor of particle

        self.bunch_r = torch.tensor([])      # Bunch position
        self.bunch_p = torch.tensor([])      # Bunch momentum
        self.bunch_beta = torch.tensor([])   # Bunch beta
        self.bunch_gamma = torch.tensor([])  # Bunch lorentz factor

        # initial beam parameters
        self.init_r0 = None
        self.init_d0 = None
        self.init_gamma = None
        self.beam_params = None
        self.beam_matrix = None
        self.c_light = 0.299792458

    def load_file(self, track_file: str) -> None:
        """
        Loads track from external simulation
        :param track_file: Numpy array containing track information. In format:
         [time, r, beta, gamma]
        """
        track = np.load(track_file)
        time = track[0]
        r = track[1:4]
        beta = track[4:]

        self.time = torch.from_numpy(time).type(torch.get_default_dtype()
                                                ).to(self.device)
        self.r = torch.from_numpy(r).type(torch.get_default_dtype()
                                          ).to(self.device)
        self.beta = torch.from_numpy(beta).type(torch.get_default_dtype()
                                                ).to(self.device)
        self.gamma = (1 - torch.sum(self.beta[:, 0]**2.))**-0.5

    @torch.jit.ignore
    def plot_track(self, axes: List[int], pos: bool = True
                   ) -> Tuple[plt.Figure, plt.Axes]:
        """
        Plot interpolated track (Uses cubic spline)
        :param axes: Axes to plot (e. g z-x [2, 0])
        :param pos: Bool if true then plot position else plot beta
        :return: fig, ax
        """
        fig, ax = plt.subplots()
        if pos:
            ax.plot(self.r[axes[0], :].cpu().detach().numpy(),
                    self.r[axes[1], :].cpu().detach().numpy())
        else:
            ax.plot(self.beta[axes[0], :].cpu().detach().numpy(),
                    self.beta[axes[1], :].cpu().detach().numpy())
        return fig, ax

    @torch.jit.ignore
    def plot_bunch(self, axes: List[int], n_part: int = -1, pos: bool = True
                   ) -> Tuple[plt.Figure, plt.Axes]:
        """
        Plot single particle track
        :param axes: Axes to plot (e. g z-x [2, 0])
        :param n_part: Number of tracks to plot. Defaults to all
        :param pos: Bool if true then plot position else plot beta
        :return: fig, ax
        """
        if self.bunch_r.shape[0] == 0:
            raise Exception("No bunch has been simulated so nothing can be "
                            "plotted.")
        fig, ax = plt.subplots()
        if pos:
            ax.plot(self.bunch_r[:n_part, axes[0], :].cpu().detach().T,
                    self.bunch_r[:n_part, axes[1], :].cpu().detach().T)
        else:
            ax.plot(self.bunch_beta[:n_part, axes[0], :].cpu().detach().T,
                    self.bunch_beta[:n_part, axes[1], :].cpu().detach().T)
        return fig, ax

    @torch.jit.export
    def switch_device(self, device: torch.device) -> "Track":
        """
        Changes the device that the class data is stored on.
        :param device: Device to switch to.
        """
        self.device = device
        if self.r.shape[0] != 0:
            self.time = self.time.to(device)
            self.r = self.r.to(device)
            self.beta = self.beta.to(device)
            self.gamma = self.gamma.to(device)
        if self.bunch_r.shape[0] != 0:
            self.bunch_r = self.bunch_r.to(device)
            self.bunch_beta = self.bunch_beta.to(device)
            self.bunch_gamma = self.bunch_gamma.to(device)
        return self

    @torch.jit.ignore
    def set_central_params(self, position: torch.Tensor,
                           direction: torch.Tensor, gamma: torch.Tensor
                           ) -> None:
        """
        Sets the central parameters for the track.
        :param position: Initial position of central particle.
        :param direction: Initial direction of central particle.
        :param gamma: Initial energy of central particle.
        """
        self.init_r0 = position
        self.init_d0 = direction
        self.init_gamma = gamma

    @torch.jit.ignore
    def set_beam_params(self, beam_moments: torch.Tensor) -> None:
        """
        Sets the beam moments for the track.
        :param beam_moments: Beam moments [<x-x0>^2, <x-x0><x'-x0'>, <x'-x0'>^2,
                                           <y-y0>^2, <y-y0><y'-y0'>, <y'-y0'>^2,
                                           <z-z0>^2, <g-g0>^2]
        """
        if beam_moments.shape[0] != 8:
            raise Exception("Beam moments must be of shape [8,]")
        self.beam_params = beam_moments
        self.beam_matrix = torch.zeros((6, 6), device=self.device)
        self.beam_matrix[0, 0] = beam_moments[0]
        self.beam_matrix[0, 1] = beam_moments[1]
        self.beam_matrix[1, 0] = beam_moments[1]
        self.beam_matrix[1, 1] = beam_moments[2]
        self.beam_matrix[2, 2] = beam_moments[3]
        self.beam_matrix[2, 3] = beam_moments[4]
        self.beam_matrix[3, 2] = beam_moments[4]
        self.beam_matrix[3, 3] = beam_moments[5]
        self.beam_matrix[4, 4] = beam_moments[6]
        self.beam_matrix[5, 5] = beam_moments[7]

    @torch.jit.export
    def get_beam_matrix(self) -> torch.Tensor:
        """
        Returns the beam matrix.
        :return: Beam matrix.
        """
        if self.beam_matrix is None:
            raise Exception("Beam matrix has not been set.")
        return self.beam_matrix

    @torch.jit.export
    def sim_central(self, time: torch.Tensor) -> None:
        """
        Models the trajectory of a single particle through a field defined
        by field_container
        :param time: Array of time samples.
        """

        if self.field_container is None:
            raise TypeError("Field container value is None.")

        if (self.init_r0 is None or self.init_d0 is None
                or self.init_gamma is None):
            raise TypeError("Initial parameters have not been set.")

        if time.shape[0] % 2 == 0:
            print(f"Warning: Filon integrator is a Simpson's based method"
                  f" requiring an odd number of time steps for high accuracy."
                  f" Increasing steps to {time.shape[0] + 1}")
            self.time = torch.linspace(time[0], time[-1], time.shape[0] + 1,
                                       device=time.device)
        else:
            self.time = time

        self.r = torch.zeros((self.time.shape[0], 3), device=self.device)
        self.p = torch.zeros((self.time.shape[0], 3), device=self.device)
        self.gamma = self.init_gamma

        self.r[0] = self.init_r0
        self.p[0] = (self.c_light * (self.init_gamma**2.0 - 1.)**0.5
                     * self.init_d0 / torch.norm(self.init_d0))

        for i, t in enumerate(self.time[:-1]):
            detla_t = (self.time[i+1] - self.time[i])
            delta_t_2 = detla_t / 2.

            field = self.field_container.get_field(self.r[i].clone())
            r_k1 = self._dr_dt(self.p[i].clone())
            p_k1 = self._dp_dt(self.p[i].clone(), field)

            field = self.field_container.get_field(self.r[i].clone() + r_k1
                                                   * delta_t_2)
            r_k2 = self._dr_dt(self.p[i].clone() + p_k1 * delta_t_2)
            p_k2 = self._dp_dt(self.p[i].clone() + p_k1 * delta_t_2, field)

            field = self.field_container.get_field(self.r[i].clone() + r_k2
                                                   * delta_t_2)
            r_k3 = self._dr_dt(self.p[i].clone() + p_k2 * delta_t_2)
            p_k3 = self._dp_dt(self.p[i].clone() + p_k2 * delta_t_2, field)

            field = self.field_container.get_field(self.r[i].clone() + r_k3
                                                   * detla_t)
            r_k4 = self._dr_dt(self.p[i].clone() + p_k3 * detla_t)
            p_k4 = self._dp_dt(self.p[i].clone() + p_k3 * detla_t, field)

            self.r[i+1] = torch.squeeze(self.r[i].clone() + (detla_t / 6.)
                                        * (r_k1 + 2. * r_k2 + 2. * r_k3 + r_k4))
            self.p[i+1] = torch.squeeze(self.p[i].clone() + (detla_t / 6.)
                                        * (p_k1 + 2. * p_k2 + 2. * p_k3 + p_k4))

        self.beta = self.p / (self.c_light**2.0
                              + torch.sum(self.p * self.p, dim=1)[:, None])**0.5
        # Transpose for field solver and switch device
        self.r = self.r.to(self.device).T
        self.beta = self.beta.to(self.device).T
        self.time = self.time.to(self.device)

    @torch.jit.export
    def sim_beam(self, time: torch.Tensor, n_part: Optional[int] = None,
                 bunch_r: Optional[torch.Tensor] = None,
                 bunch_d: Optional[torch.Tensor] = None,
                 bunch_g: Optional[torch.Tensor] = None) -> None:
        """
        Models the trajectory of a bunch of particle through a field defined
        by field_container (cpp version should be much-much faster). If bunch_r,
        bunch_d and bunch_gamma are provided then they are used instead of the
        beam moments from set_beam_params.
        :param time: Array of time samples.
        :param n_part: Number of particles to simulate.
        :param bunch_r: Initial position of tracks.
        :param bunch_d: Initial direction of tracks.
        :param bunch_g: Initial lorentz factor tracks.
        """

        if time.shape[0] % 2 == 0:
            print(f"Warning: Filon integrator is a Simpson's based method"
                  f" requiring an odd number of time steps for high accuracy."
                  f" Increasing steps to {time.shape[0] + 1}")
            self.time = torch.linspace(time[0], time[-1], time.shape[0] + 1,
                                       device=self.device)
        else:
            self.time = time.to(self.device)

        # if no bunch parameters are provided then use the ones from
        # set_beam_params
        if bunch_r is None or bunch_d is None or bunch_g is None:
            if self.beam_params is None:
                raise TypeError("Beam parameters have not been set.")
            else:
                bunch_r, bunch_d, bunch_gamma = self._sample_beam(n_part)

        # make sure all arrays are the same shape
        samples = bunch_r.shape[0]
        self.bunch_r = torch.zeros((samples, self.time.shape[0], 3),
                                   device=self.device)
        self.bunch_p = torch.zeros((samples, self.time.shape[0], 3),
                                   device=self.device)
        self.bunch_r[:, 0] = bunch_r
        self.bunch_p[:, 0] = self.c_light * (bunch_g[:, None]**2. - 1)**0.5\
                             * bunch_d / torch.norm(bunch_d, dim=1)[:, None]

        for i, t in enumerate(time[:-1]):
            detla_t = (time[i+1] - time[i])
            delta_t_2 = detla_t / 2.

            field = self.field_container.get_field(self.bunch_r[:, i].clone())
            r_k1 = self._dr_dt(self.bunch_p[:, i].clone())
            p_k1 = self._dp_dt(self.bunch_p[:, i].clone(), field)

            field = self.field_container.get_field(self.bunch_r[:, i].clone()
                                                   + r_k1 * delta_t_2)
            r_k2 = self._dr_dt(self.bunch_p[:, i].clone() + p_k1 * delta_t_2)
            p_k2 = self._dp_dt(self.bunch_p[:, i].clone() + p_k1 * delta_t_2,
                               field)

            field = self.field_container.get_field(self.bunch_r[:, i].clone()
                                                   + r_k2 * delta_t_2)
            r_k3 = self._dr_dt(self.bunch_p[:, i].clone() + p_k2 * delta_t_2)
            p_k3 = self._dp_dt(self.bunch_p[:, i].clone() + p_k2 * delta_t_2,
                               field)

            field = self.field_container.get_field(self.bunch_r[:, i].clone()
                                                   + r_k3 * detla_t)
            r_k4 = self._dr_dt(self.bunch_p[:, i].clone() + p_k3 * detla_t)
            p_k4 = self._dp_dt(self.bunch_p[:, i].clone() + p_k3 * detla_t,
                               field)

            self.bunch_r[:, i+1] = self.bunch_r[:, i].clone() + (detla_t / 6.) \
                                   * (r_k1 + 2. * r_k2 + 2. * r_k3 + r_k4)
            self.bunch_p[:, i+1] = self.bunch_p[:, i].clone() + (detla_t / 6.) \
                                   * (p_k1 + 2. * p_k2 + 2. * p_k3 + p_k4)

        self.bunch_beta = self.bunch_p / (self.c_light**2.0 + torch.sum(
            self.bunch_p * self.bunch_p, dim=-1)[..., None])**0.5

        self.r = torch.mean(self.bunch_r, dim=0).T
        self.beta = torch.mean(self.bunch_beta, dim=0).T
        self.gamma = torch.mean(bunch_g, dim=0)
        self.bunch_r = self.bunch_r.permute((0, 2, 1))
        self.bunch_beta = self.bunch_beta.permute((0, 2, 1))
        self.bunch_gamma = bunch_g.to(self.device)

    @torch.jit.ignore
    def sim_central_c(self, time: torch.Tensor) -> None:
        """
        Models the trajectory of a single particle through a field defined
        by field_container is cpp version (should be much-much faster)
        :param time: Array of times
        """

        if self.field_container is None:
            raise TypeError("Field container value is None.")

        if (self.init_r0 is None or self.init_d0 is None
                or self.init_gamma is None):
            raise TypeError("Initial parameters have not been set.")

        if time.shape[0] % 2 == 0:
            print(f"Warning: Filon integrator is a Simpson's based method"
                  f" requiring an odd number of time steps for high accuracy."
                  f" Increasing steps to {time.shape[0] + 1}")
            steps = time.shape[0] + 1
        else:
            steps = time.shape[0]

        r0_c = cTrack.ThreeVector(self.init_r0[0], self.init_r0[1],
                                  self.init_r0[2])
        d0_c = cTrack.ThreeVector(self.init_d0[0], self.init_d0[1],
                                  self.init_d0[2])
        field = self.field_container.gen_c_container()
        track = cTrack.Track()
        track.setTime(time[0], time[-1], steps)
        track.setCentralInit(r0_c, d0_c, self.init_gamma.item())
        track.setField(field)
        time, r, beta = track.simulateTrack()

        # Transpose for field solver and switch device
        self.time = torch.from_numpy(time).type(torch.get_default_dtype()
                                                ).to(self.device)
        self.r = torch.from_numpy(r).type(torch.get_default_dtype()
                                          ).to(self.device).T
        self.beta = torch.from_numpy(beta).type(torch.get_default_dtype()
                                                ).to(self.device).T
        self.gamma = torch.tensor(
            [self.init_gamma.item()], dtype=torch.get_default_dtype()
            ).to(self.device)

    @torch.jit.ignore
    def sim_beam_c(self, n_part: int, time: torch.Tensor) -> None:
        """
        Models the trajectory of a single particle through a field defined
        by field_container with cpp version (should be much-much faster)
        :param n_part: Number of particles to simulate.
        :param time: Array of time samples.
        """
        if self.field_container is None:
            raise TypeError("Field container value is None.")

        if (self.init_r0 is None or self.init_d0 is None
                or self.init_gamma is None):
            raise TypeError("Initial parameters have not been set.")

        if self.beam_params is None:
            raise TypeError("Beam parameters have not been set.")

        # TODO allow for self sampled tracks to be simulated

        if time.shape[0] % 2 == 0:
            print(f"Warning: Filon integrator is a Simpson's based method"
                  f" requiring an odd number of time steps for high accuracy."
                  f" Increasing steps to {time.shape[0] + 1}")
            time = torch.linspace(time[0], time[-1], time.shape[0] + 1)

        # First we simulate the central track
        self.sim_central_c(time)
        r0_c = cTrack.ThreeVector(self.init_r0[0], self.init_r0[1],
                                  self.init_r0[2])
        d0_c = cTrack.ThreeVector(self.init_d0[0], self.init_d0[1],
                                  self.init_d0[2])
        field = self.field_container.gen_c_container()
        track = cTrack.Track()
        track.setTime(time[0], time[-1], time.shape[0])
        track.setCentralInit(r0_c, d0_c, self.init_gamma.item())
        track.setBeamParams(self.beam_params)
        track.setField(field)
        time, r, beta = track.simulateBeam(n_part)

        # Transpose for field solver and switch device
        self.bunch_r = torch.from_numpy(r.transpose((0, 2, 1))).type(
            torch.get_default_dtype()).to(self.device)
        self.bunch_beta = torch.from_numpy(beta.transpose((0, 2, 1))).type(
            torch.get_default_dtype()).to(self.device)
        self.bunch_gamma = 1. / (1. - torch.sum(self.bunch_beta[:, :, 0]**2.,
                                                dim=1))**0.5

    def _sample_beam(self, n_part: int) -> Tuple[torch.Tensor, torch.Tensor,
                                                 torch.Tensor]:
        """
        Samples the beam using the beam moments. Beam moments must be set
        before calling this function.
        :param n_part: Number of particles to sample.
        """
        if (self.beam_params is None or self.init_r0 is None
                or self.init_d0 is None or self.init_gamma is None):
            raise TypeError("Beam moments or central parameters have not been"
                            " set.")

        # First set up the distribution classes
        x_angle = torch.arctan(self.init_d0[0] / self.init_d0[2])
        x_dist = MultivariateNormal(
            loc=torch.tensor([self.init_r0[0], x_angle]),
            covariance_matrix=torch.tensor([[
                self.beam_params[0], self.beam_params[1]],
                [self.beam_params[1], self.beam_params[2]]]))
        y_angle = torch.arctan(self.init_d0[1] / self.init_d0[2])
        y_dist = MultivariateNormal(
            loc=torch.tensor([self.init_r0[1], y_angle]),
            covariance_matrix=torch.tensor([[
                self.beam_params[0], self.beam_params[1]],
                [self.beam_params[1], self.beam_params[2]]]))
        g_dist = Normal(loc=self.init_gamma, scale=self.beam_params[7]**0.5)
        x_sample = x_dist.sample((n_part,))
        y_sample = y_dist.sample((n_part,))
        bunch_gamma = g_dist.sample((n_part,))
        beam_r0 = torch.stack([x_sample[:, 0], y_sample[:, 0],
                               torch.ones_like(x_sample[:, 0])
                               * self.init_r0[2]], dim=1)
        beam_d0 = torch.stack([torch.arctan(x_sample[:, 1]),
                               torch.arctan(y_sample[:, 1]),
                               torch.ones_like(x_sample[:, 0])], dim=1)
        return beam_r0, beam_d0, bunch_gamma

    def _dp_dt(self, p: torch.Tensor, field: torch.Tensor) -> torch.Tensor:
        """
        Rate of change of beta w.r.t time. We assume acceleration is always
        perpendicular to velocity which is true for just magnetic field
        :param p: Particle momentum.
        :param field: Magnetic field.
        :return: Momentum gradient
        """
        gamma = torch.atleast_1d(1.0 + torch.sum(p * p, dim=-1)
                                 / self.c_light**2.0)**0.5
        return -1 * torch.cross(p, field, dim=-1) / gamma[:, None]

    def _dr_dt(self, p: torch.Tensor) -> torch.Tensor:
        """
        Rate of change of position w.r.t time
        :param p: Particle momentum
        :return: position gradient
        """
        gamma = torch.atleast_1d(1.0 + torch.sum(p * p, dim=-1) /
                                 self.c_light**2.0)**0.5
        return p / gamma[:, None]
