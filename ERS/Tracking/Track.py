import numpy as np
import torch
import torch.linalg
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
        if self.bunch_r.shape[0] != 0:
            self.bunch_r = self.bunch_r.to(device)
            self.bunch_beta = self.bunch_beta.to(device)
        return self

    @torch.jit.export
    def sim_single(self, time: torch.Tensor, r_0: torch.Tensor,
                   d_0: torch.Tensor, gamma: torch.Tensor) -> None:
        """
        Models the trajectory of a single particle through a field defined
        by field_container
        :param time: Array of time samples.
        :param r_0: Initial position of particle
        :param d_0: Initial direction of particle
        :param gamma: Initial lorentz factor of particle
        """
        # TODO add some checks to make sure setup is ok. e.g. check that
        #  starting position is inside first field element

        if self.field_container is None:
            raise TypeError("Field container value is None.")

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
        self.gamma = gamma

        detla_t = (self.time[1] - self.time[0])
        delta_t_2 = detla_t / 2.

        self.r[0] = r_0
        self.p[0] = self.c_light * (gamma**2.0 - 1.)**0.5 * d_0\
                    / torch.norm(d_0)

        for i, t in enumerate(self.time[:-1]):
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

            self.r[i+1] = torch.squeeze(self.r[i].clone() + (detla_t / 6.) \
                          * (r_k1 + 2. * r_k2 + 2. * r_k3 + r_k4))
            self.p[i+1] = torch.squeeze(self.p[i].clone() + (detla_t / 6.) \
                          * (p_k1 + 2. * p_k2 + 2. * p_k3 + p_k4))

        self.beta = self.p / (self.c_light**2.0
                              + torch.sum(self.p * self.p, dim=1)[:, None])**0.5
        # Transpose for field solver and switch device
        self.r = self.r.to(self.device).T
        self.beta = self.beta.to(self.device).T
        self.time = self.time.to(self.device)

    @torch.jit.export
    def sim_bunch(self, bunch_r: torch.Tensor,
                  bunch_d: torch.Tensor, bunch_gamma: torch.Tensor,
                  time: torch.Tensor) -> None:
        """
        Models the trajectory of a bunch of particle through a field defined
        by field_container is cpp version (should be much-much faster)
        :param bunch_r: Initial position of tracks.
        :param bunch_d: Initial direction of tracks.
        :param bunch_gamma: Initial lorentz factor tracks.
        :param time: Array of time samples.
        """

        if time.shape[0] % 2 == 0:
            print(f"Warning: Filon integrator is a Simpson's based method"
                  f" requiring an odd number of time steps for high accuracy."
                  f" Increasing steps to {time.shape[0] + 1}")
            self.time = torch.linspace(time[0], time[-1], time.shape[0] + 1,
                                       device=self.device)
        else:
            self.time = time.to(self.device)

        # make sure all arrays are the same shape
        samples = bunch_r.shape[0]
        self.bunch_r = torch.zeros((samples, self.time.shape[0], 3),
                                   device=self.device)
        self.bunch_p = torch.zeros((samples, self.time.shape[0], 3),
                                   device=self.device)
        detla_t = (time[1] - time[0])
        delta_t_2 = detla_t / 2.
        self.bunch_r[:, 0] = bunch_r
        self.bunch_p[:, 0] = self.c_light * (bunch_gamma[:, None]**2. - 1)**0.5\
                             * bunch_d / torch.norm(bunch_d, dim=1)[:, None]

        for i, t in enumerate(time[:-1]):
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
        self.gamma = torch.mean(bunch_gamma, dim=0)
        self.bunch_r = self.bunch_r.permute((0, 2, 1))
        self.bunch_beta = self.bunch_beta.permute((0, 2, 1))
        self.bunch_gamma = bunch_gamma.to(self.device)

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

    @torch.jit.unused
    def sim_single_c(self, time: torch.Tensor, r_0: torch.Tensor,
                     d_0: torch.Tensor, gamma: float) -> None:
        """
        Models the trajectory of a single particle through a field defined
        by field_container is cpp version (should be much-much faster)
        :param time: Array of times
        :param r_0: Initial position of particle
        :param d_0: Initial direction of particle
        :param gamma: Initial lorentz factor of particle
        """

        if self.field_container is None:
            raise TypeError("Field container value is None.")

        if time.shape[0] % 2 == 0:
            print(f"Warning: Filon integrator is a Simpson's based method"
                  f" requiring an odd number of time steps for high accuracy."
                  f" Increasing steps to {time.shape[0] + 1}")
            steps = time.shape[0] + 1
        else:
            steps = time.shape[0]

        r0_c = cTrack.ThreeVector(r_0[0], r_0[1], r_0[2])
        d0_c = cTrack.ThreeVector(d_0[0], d_0[1], d_0[2])
        field = self.field_container.gen_c_container()
        track = cTrack.Track()
        track.setTime(time[0], time[-1], steps)
        track.setCentralInit(r0_c, d0_c, gamma)
        track.setField(field)
        time, r, beta = track.simulateTrack()

        # Transpose for field solver and switch device
        self.time = torch.from_numpy(time).type(torch.get_default_dtype()
                                                ).to(self.device)
        self.r = torch.from_numpy(r).type(torch.get_default_dtype()
                                          ).to(self.device).T
        self.beta = torch.from_numpy(beta).type(torch.get_default_dtype()
                                                ).to(self.device).T
        self.gamma = torch.tensor([gamma], dtype=torch.get_default_dtype()
                                  ).to(self.device)

    @torch.jit.unused
    def sim_bunch_c(self, n_part: int, time: torch.Tensor, r_0: torch.Tensor,
                    d_0: torch.Tensor, gamma: float, bunch_params: torch.Tensor
                    ) -> None:
        """
        Models the trajectory of a single particle through a field defined
        by field_container with cpp version (should be much-much faster)
        :param n_part: Number of particles to simulate.
        :param time: Array of time samples.
        :param r_0: Initial position of central track.
        :param d_0: Initial direction of central track.
        :param gamma: Initial lorentz factor of central track.
        :param bunch_params: torch.Tensor of 2nd order moments in the format:
         [sig_x, sig_x_xp, sig_xp, sig_x, sig_y_yp, sig_yp, sig_gamma]
        """
        if self.field_container is None:
            raise TypeError("Field container value is None.")

        if time.shape[0] % 2 == 0:
            print(f"Warning: Filon integrator is a Simpson's based method"
                  f" requiring an odd number of time steps for high accuracy."
                  f" Increasing steps to {time.shape[0] + 1}")
            time = torch.linspace(time[0], time[-1], time.shape[0] + 1)

        # First we simulate the central track
        self.sim_single_c(time, r_0, d_0, gamma)

        r0_c = cTrack.ThreeVector(r_0[0], r_0[1], r_0[2])
        d0_c = cTrack.ThreeVector(d_0[0], d_0[1], d_0[2])
        field = self.field_container.gen_c_container()
        track = cTrack.Track()
        track.setTime(time[0], time[-1], time.shape[0])
        track.setCentralInit(r0_c, d0_c, gamma)
        track.setBeamParams(bunch_params)
        track.setField(field)
        time, r, beta = track.simulateBeam(n_part)

        # Transpose for field solver and switch device
        self.bunch_r = torch.from_numpy(r.transpose((0, 2, 1))).type(
            torch.get_default_dtype()).to(self.device)
        self.bunch_beta = torch.from_numpy(beta.transpose((0, 2, 1))).type(
            torch.get_default_dtype()).to(self.device)
        self.bunch_gamma = 1. / (1. - torch.sum(self.bunch_beta[:, :, 0]**2.,
                                                dim=1))**0.5
