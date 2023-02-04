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
        self.beta = None  # Velocity along particle path

        self.field_container = None
        self.field_index = None

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
            ax.plot(self.r[:, axes[0]].cpu().detach().numpy(),
                    self.r[:, axes[1]].cpu().detach().numpy())
        else:
            ax.plot(self.beta[:, axes[0]].cpu().detach().numpy(),
                    self.beta[:, axes[1]].cpu().detach().numpy())
        return fig, ax

    def sim_single(self, field_container, time, r_0, beta_0):
        """
        Models the trajectory of a single particle through a field defined
        by field_container
        :param field_container: Instance of class FieldContainer.
        :param time: Array of times
        :param r_0:
        :param beta_0:
        """
        # TODO add some checks to make sure setup is ok. e.g. check that
        #  starting position is inside first field element
        detla_t = (time[1] - time[0])
        delta_t_2 = detla_t / 2.
        self.time = time
        self.r = torch.zeros((self.time.shape[0], 3), device=self.device)
        self.beta = torch.zeros((self.time.shape[0], 3), device=self.device)
        self.field_index = torch.zeros_like(time, dtype=torch.int,
                                            device=self.device)
        self.r[0] = r_0
        self.beta[0] = beta_0
        self.field_index[0] = 0

        for i, t in enumerate(time[:-1]):
            self.field_index[i] = field_container.get_index(self.r[i, 2],
                                                        self.field_index[i-1])
            field = field_container.get_field(self.r[i], self.field_index[i])
            r_k1 = self.beta[i]
            beta_k1 = self._db_dt(r_k1, field)

            field = field_container.get_field(self.r[i] + c_light * r_k1
                                              * delta_t_2, self.field_index[i])
            r_k2 = self.beta[i] + beta_k1 * delta_t_2
            beta_k2 = self._db_dt(r_k2, field)

            field = field_container.get_field(self.r[i] + c_light * r_k2
                                              * delta_t_2, self.field_index[i])
            r_k3 = self.beta[i] + beta_k2 * delta_t_2
            beta_k3 = self._db_dt(r_k3, field)

            field = field_container.get_field(self.r[i] + c_light * r_k3
                                              * detla_t, self.field_index[i])
            r_k4 = self.beta[i] + beta_k3 * detla_t
            beta_k4 = self._db_dt(r_k4, field)
            self.r[i+1] = self.r[i] + (detla_t * c_light / 6.) \
                          * (r_k1 + 2. * r_k2 + 2. * r_k3 + r_k4)
            self.beta[i+1] = self.beta[i] + (detla_t / 6.) \
                             * (beta_k1 + 2. * beta_k2 + 2. * beta_k3 + beta_k4)

    @staticmethod
    def _db_dt(beta, field):
        """
        Rate of change of beta w.r.t time. We assume acceleration is always
        perpendicular to velocity which is true for just magnetic field
        :param beta: Particle velocity
        :param field: Magnetic field
        :return: beta gradient
        """
        return -(1. - torch.sum(beta * beta))**0.5 * torch.cross(beta, field)


class FieldBlock:
    """
    Base class of magnetic fields.
    """
    # TODO need to add the direction of the magnet
    def __init__(self, center_pos, length, B0, direction=None, edge_length=0.,
                 field_extent=5.):
        """
        :param center_pos: Central position of the magnet.
        :param length: Length of main part of field.
        :param B0: Field parameter vector [Units are tesla and meters.]
        :param direction: Vector through central axis of magnet [0, 0, 1]
            by defult
        :param edge_length: Length for field to fall by 10 %. 0 for hard edge or
            > 0 for soft edge with B0 / (1 + (z / d)**2)**2 dependence.
        :param field_extent: Extent of field beyond the main part. Given in
            units of edge_length.
        """
        self.center_pos = torch.tensor(center_pos)
        self.length = length
        self.B0 = B0 / (me / (qe * 1.e-9))
        self.direction = direction
        self.edge_scaled = edge_length / (10**0.5 - 1)**0.5
        self.field_extent = field_extent * edge_length
        self.bounds = [center_pos[2] - 0.5 * self.length - self.field_extent,
                       center_pos[2] + 0.5 * self.length + self.field_extent]

    def get_field(self, position):
        """
        Gets the field at a given location.
        :param position: Position of particle
        :return: Magnetic field vector.
        """
        pass

    def in_field(self, z):
        """
        Check if given z position is in field
        :param z: Longitudinal position
        :return: True if inside or false if outside.
        """
        return (self.center_pos - self.length - self.field_extent
                < z < self.center_pos + self.length + self.field_extent)

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

    def __init__(self, center_pos, length, B0, direction=None, edge_length=0.,
                 field_extent=5.):
        """
        :param center_pos: Central position of the magnet.
        :param length: Length of main part of field.
        :param B0: Field strength in y direction [0, B0, 0]
        :param direction: Vector through central axis of magnet [0, 0, 1]
            by defult
        :param edge_length: Length for field to fall by 10 %. 0 for hard edge or
            > 0 for soft edge with B0 / (1 + (z / d)**2)**2 dependence.
        :param field_extent: Extent of field beyond the main part. Given in
            units of edge_length.
        """
        super().__init__(center_pos, length, torch.tensor([0, B0, 0]),
                         direction, edge_length, field_extent)

    def get_field(self, position):
        """
        Gets the field at a given location.
        :param position: Position of particle
        :return: Magnetic field vector.
        """
        local_pos = position - self.center_pos
        print(position, local_pos)
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
        :param field_array: A list containing all the definied elements.
        """
        # make sure fields are in order of z location
        self.field_array = sorted(field_array,
                                  key=lambda field: field.center_pos[2])

        # Create a boundary array / index
        self.bounds = []
        self.field_idx = [-1]
        for i, field in enumerate(field_array):
            self.bounds.extend(field.bounds)
            self.field_idx.extend([i, -1])
        self.bounds.append(float("inf"))

        # Check that field bounds are monotonically increasing
        for i, bound in enumerate(self.bounds[1:], 1):
            if bound <= self.bounds[i-1]:
                raise Exception("Error: Field boundaries are overlapping. "
                                "Try reducing field_extent to prevent this.")

    def get_index(self, z_pos, current_index):
        """
        Checks if particle has propagated boundary to next element
        :param z_pos: Current z position of particle
        :param current_index: Current index of particle
        :return: Index of field element we are in
        """
        if z_pos > self.bounds[current_index]:
            return current_index+1
        else:
            return current_index

    def get_field(self, position, index):
        """
        Finds which element we are in and returns field
        :param position: Position of particle
        :param index: Which element field we are in
        :return: Magnetic field vector.
        """
        if self.field_idx[index] == -1:
            return torch.tensor([0., 0., 0.])
        else:
            print(self.field_idx[index])
            return self.field_array[self.field_idx[index]].get_field(position)


if __name__ == "__main__":
    q1 = Dipole([0, 0, 0], 1, 0.1, None, 0.05, 5)
    q2 = Dipole([0, 0, 2], 1, -0.1, None, 0.05, 5)
    test = FieldContainer([q1, q2])

    gamma = 100

    track = Track()
    beta0 = torch.tensor([0., 0, 1.]) * (1 - 1./gamma**2.)**0.5
    r0 = torch.tensor([0., 0., -1.])
    time = torch.linspace(0, 15, 100)
    track.sim_single(test, time, r0, beta0)
    track.plot_track([2, 0], True)
    print(track.r[:, 2])


    plt.show()

