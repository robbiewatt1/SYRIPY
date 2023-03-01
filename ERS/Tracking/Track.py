import numpy as np
import torch
import torch.linalg
import matplotlib.pyplot as plt
import os
from .cTrack import cTrack

c_light = 0.29979245


class Track(torch.nn.Module):
    """
    Class which handles tracking of an electron through the magnetic setup. This
    can either be solved using an RK4 method or loaded from an external file.
    """

    def __init__(self, device=None):
        """
        :param device: Device being used (cpu / gpu)
        """
        super().__init__()
        # Load from track file
        self.device = device
        self.time = None  # Proper time along particle path
        self.r = None     # Particle position
        self.p = None     # Particle momentum
        self.beta = None  # Velocity along particle path

        self.bunch_time = None  # Bunch time samples
        self.bunch_r = None  # bunch position
        self.bunch_beta = None  # Bunch beta

    def load_file(self, track_file):
        """
        Loads track from external simulation
        :param track_file: Numpy array containing track information. In format:
         [time, r, beta]
        """
        track = np.load(track_file)
        time = track[0]
        r = track[1:4]
        beta = track[4:]
        self.time = torch.tensor(time, device=self.device)
        self.r = torch.tensor(r, device=self.device)
        self.beta = torch.tensor(beta, device=self.device)

    def plot_track(self, axes, pos=True):
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

    def switch_device(self, device):
        """
        Changes the device that the class data is stored on.
        :param device: Device to switch to.
        """
        self.device = device
        if self.r is not None:
            self.time = self.time.to(device)
            self.r = self.r.to(device)
            self.beta = self.beta.to(device)
        if self.bunch_r is not None:
            self.bunch_time = self.bunch_time.to(device)
            self.bunch_r = self.bunch_r.to(device)
            self.bunch_beta = self.bunch_beta.to(device)
        return self

    def sim_single(self, field_container, time, r_0, d_0, gamma):
        """
        Models the trajectory of a single particle through a field defined
        by field_container
        :param field_container: Instance of class FieldContainer.
        :param time: Array of times
        :param r_0:
        :param d_0: Initial direction of particle
        :param gamma: Initial lorentz factor of particle
        """
        # TODO add some checks to make sure setup is ok. e.g. check that
        #  starting position is inside first field element
        self.time = time
        self.r = torch.zeros((self.time.shape[0], 3))
        self.p = torch.zeros((self.time.shape[0], 3))

        detla_t = (time[1] - time[0])
        delta_t_2 = detla_t / 2.

        self.r[0] = r_0
        self.p[0] = c_light * (gamma**2.0 - 1.)**0.5 * d_0 / torch.norm(d_0)

        for i, t in enumerate(time[:-1]):
            field = field_container.get_field(self.r[i])

            r_k1 = self._dr_dt(self.p[i])
            p_k1 = self._dp_dt(self.p[i], field)

            field = field_container.get_field(self.r[i] + r_k1 * delta_t_2)
            r_k2 = self._dr_dt(self.p[i] + p_k1 * delta_t_2)
            p_k2 = self._dp_dt(self.p[i] + p_k1 * delta_t_2, field)

            field = field_container.get_field(self.r[i] + r_k2 * delta_t_2)
            r_k3 = self._dr_dt(self.p[i] + p_k2 * delta_t_2)
            p_k3 = self._dp_dt(self.p[i] + p_k2 * delta_t_2, field)

            field = field_container.get_field(self.r[i] + r_k3 * detla_t)
            r_k4 = self._dr_dt(self.p[i] + p_k3 * detla_t)
            p_k4 = self._dp_dt(self.p[i] + p_k3 * detla_t, field)

            self.r[i+1] = self.r[i] + (detla_t / 6.) \
                          * (r_k1 + 2. * r_k2 + 2. * r_k3 + r_k4)
            self.p[i+1] = self.p[i] + (detla_t / 6.) \
                             * (p_k1 + 2. * p_k2 + 2. * p_k3 + p_k4)

        self.beta = self.p / (c_light**2.0 + torch.sum(self.p * self.p,
                                                       dim=1)[:, None])**0.5

        # Transpose for field solver and switch device
        self.r = self.r.to(self.device).T
        self.beta = self.beta.to(self.device).T
        self.time = self.time.to(self.device)

    @staticmethod
    def _dp_dt(p, field):
        """
        Rate of change of beta w.r.t time. We assume acceleration is always
        perpendicular to velocity which is true for just magnetic field
        :param beta: Particle velocity
        :param field: Magnetic field
        :return: beta gradient
        """
        gamma = (1.0 + torch.sum(p * p) / c_light**2.0)**0.5
        return -1 * torch.cross(p, field) / gamma

    @staticmethod
    def _dr_dt(p):
        """
        Rate of change of position w.r.t time
        :param p: Particle momentum
        :return: position gradient
        """
        gamma = (1.0 + torch.sum(p * p) / c_light**2.0)**0.5
        return p / gamma

    def sim_single_c(self, field_container, time, r_0, d_0, gamma):
        """
        Models the trajectory of a single particle through a field defined
        by field_container is cpp version (should be much much fatser)
        :param field_container: Instance of class FieldContainer.
        :param time: Array of times
        :param r_0: Initial position of particle
        :param d_0: Initial direction of particle
        :param gamma: Initial lorentz factor of particle
        """
        self.time = time
        r0_c = cTrack.ThreeVector(r_0[0], r_0[1], r_0[2])
        d0_c = cTrack.ThreeVector(d_0[0], d_0[1], d_0[2])
        field = field_container.gen_c_container()
        track = cTrack.Track()
        track.setTime(time[0], time[-1], time.shape[0])
        track.setCentralInit(r0_c, d0_c, gamma)
        track.setField(field)
        time, r, beta = track.simulateTrack()

        # Transpose for field solver and switch device
        self.time = torch.tensor(time).to(self.device)
        self.r = torch.tensor(r).to(self.device).T
        self.beta = torch.tensor(beta).to(self.device).T

    def sim_bunch_c(self, n_part, field_container, time, r_0, d_0, gamma,
                    bunch_params):
        """
        Models the trajectory of a single particle through a field defined
        by field_container is cpp version (should be much much faster)
        :param n_part: Number of particles to simulate
        :param field_container: Instance of class FieldContainer.
        :param time: Array of times
        :param r_0:
        :param d_0: Initial direction of particle
        :param gamma: Initial lorentz factor of particle
        :param bunch_params: np.array of 2nd order moments in the format:
         [sig_x, sig_x_xp, sig_xp, sig_x, sig_y_yp, sig_yp, sig_gamma]
        """

        # First we simulate the central track
        self.sim_single_c(field_container, time, r_0, d_0, gamma)

        self.bunch_time = time
        r0_c = cTrack.ThreeVector(r_0[0], r_0[1], r_0[2])
        d0_c = cTrack.ThreeVector(d_0[0], d_0[1], d_0[2])
        field = field_container.gen_c_container()
        track = cTrack.Track()
        track.setTime(time[0], time[-1], time.shape[0])
        track.setCentralInit(r0_c, d0_c, gamma)
        track.setBeamParams(bunch_params)
        track.setField(field)
        time, r, beta = track.simulateBeam(n_part)

        # TODO Check that shape is right

        # Transpose for field solver and switch device
        self.bunch_time = torch.tensor(self.bunch_time).to(self.device)
        self.bunch_r = torch.tensor(r.transpose((0, 2, 1))).to(self.device)
        self.bunch_beta = torch.tensor(beta.transpose((0, 2, 1))).to(
            self.device)
