import numpy as np
import torch
import torch.linalg
import matplotlib.pyplot as plt

me = 9.1093837e-31
qe = 1.60217663e-19
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
        self.p = None
        self.beta = None  # Velocity along particle path

        self.field_container = None

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


class FieldBlock:
    """
    Base class of magnetic fields.
    """
    # TODO need to add the direction of the magnet
    def __init__(self, center_pos, length, B0, direction=None, edge_length=0.):
        """
        :param center_pos: Central position of the magnet.
        :param length: Length of main part of field.
        :param B0: Field parameter vector [Units are tesla and meters.]
        :param direction: Vector through central axis of magnet [0, 0, 1]
            by defult
        :param edge_length: Length for field to fall by 10 %. 0 for hard edge or
            > 0 for soft edge with B0 / (1 + (z / d)**2)**2 dependence.
        """
        self.center_pos = torch.tensor(center_pos)
        self.length = length
        self.B0 = B0 / (me / (qe * 1.e-9))
        self.direction = direction
        self.edge_scaled = edge_length / (10.**0.5 - 1)**0.5

    def get_field(self, position):
        """
        Gets the field at a given location.
        :param position: Position of particle
        :return: Magnetic field vector.
        """
        pass

    def _fridge(self, b, z):
        """
        Calculates the fringe field at a given location.
        :param b: Field vector at edge.
        :param z: Distance from end of main field.
        :return: Fringe field vector [bx, by, bz] (bx always 0)
        """
        return b / (1 + (z / self.edge_scaled)**2.)**2.


class Dipole(FieldBlock):
    """
    Defines a dipole field. The field is a constant value inside the main length
    and decays as
    """

    def __init__(self, center_pos, length, B0, direction=None, edge_length=0.):
        """
        :param center_pos: Central position of the magnet.
        :param length: Length of main part of field.
        :param B0: Field strength in y direction [0, B0, 0]
        :param direction: Vector through central axis of magnet [0, 0, 1]
            by defult
        :param edge_length: Length for field to fall by 10 %. 0 for hard edge or
            > 0 for soft edge with B0 / (1 + (z / d)**2)**2 dependence.
        """
        super().__init__(center_pos, length, torch.tensor([0, B0, 0]),
                         direction, edge_length)

    def get_field(self, position):
        """
        Gets the field at a given location.
        :param position: Position of particle
        :return: Magnetic field vector.
        """
        local_pos = position - self.center_pos
        zr = torch.abs(local_pos[2]) - 0.5 * self.length
        if zr < 0:
            return self.B0
        else:
            return self._fridge(self.B0, zr)


class Quadrupole(FieldBlock):
    """
    Defines a Quadrupole field. The field increases linearly off axis. Positive
    gradient means focusing in x and negative is focusing in y.
    """

    def __init__(self, center_pos, length, gradB, direction=None,
                 edge_length=0., field_extent=5.):
        """
        :param center_pos: Central position of the magnet.
        :param length: Length of main part of field.
        :param gradB: Gradient of field strength.
        :param direction: Vector through central axis of magnet [0, 0, 1]
            by defult
        :param edge_length: Length for field to fall by 10 %. 0 for hard edge or
            > 0 for soft edge with B0 / (1 + (z / d)**2)**2 dependence.
        :param field_extent: Extent of field beyond the main part. Given in
            units of edge_length.
        """
        super().__init__(center_pos, length, torch.tensor([gradB, -gradB, 0]),
                         direction, edge_length, field_extent)

    def get_field(self, position):
        """
        Gets the field at a given location.
        :param position: Position of particle
        :return: Magnetic field vector.
        """
        local_pos = position - self.center_pos
        zr = torch.abs(local_pos[2]) - 0.5 * self.length
        if zr < 0:
            return self.B0 * local_pos[[1, 0, 2]]
        else:
            return self._fridge(self.B0 * local_pos[[1, 0, 2]], zr)


class FieldContainer:
    """
    class containing a list of all defined magnetic elements. Will return the
    field for any given location.
    """

    def __init__(self, field_array):
        """
        :param field_array: A list containing all the defined elements.
        """
        self.field_array = field_array

    def get_field(self, position):
        """
        Finds which element we are in and returns field
        :param position: Position of particle
        :return: Magnetic field vector.
        """
        field = torch.zeros_like(position)
        for element in self.field_array:
            field += element.get_field(position)
        return field


if __name__ == "__main__":
    gamma = 339.3 / 0.51099890221

    q1 = Dipole([0, 0, 0], 0.203274830142196, -0.49051235, None, 0.01)
    q2 = Dipole([0, 0, 1.0334], 0.203274830142196, -0.49051235, None,
                0.01)
    test = FieldContainer([q1, q2])

    track = Track()
    d0 = torch.tensor([0.09320325784982, 0, 1])
    r0 = torch.tensor([-0.09318194668, 0, -1])
    time = torch.linspace(0, 10, 1000)
    track.sim_single(test, time, r0, d0, gamma)
    track.plot_track([2, 0], True)
    plt.show()

