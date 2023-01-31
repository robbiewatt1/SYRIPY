import numpy as np
import torch
import matplotlib.pyplot as plt


class Track(torch.nn.Module):
    """
    Class which handles the central electron beam track.
    """

    def __init__(self, device=None):
        """
        :param device: Device being used (cpu / gpu)
        """
        super().__init__()
        # Load from track file
        self.device = device
        self.time = None
        self.r = None
        self.beta = None

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

    def plot_track(self, axes):
        """
        Plot interpolated track (Uses cubic spline)
        :param axes: Axes to plot (e. g z-x [2, 0])
        :return: fig, ax
        """
        fig, ax = plt.subplots()
        ax.plot(self.r[:, axes[0]].cpu().detach().numpy(),
                self.r[:, axes[1]].cpu().detach().numpy())
        return fig, ax

    def init_sim(self, field_container, init_pos, time):
        """
        Passes an array of magnetic fields to the
        :param field_container: Instance of field container class
        :param init_pos:
        :param time:
        :return:
        """
        pass

    def sim_track(self):
        pass


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
        :param B0: Field parameter vector.
        :param direction: Vector through central axis of magnet [0, 0, 1]
            by defult
        :param edge_length: Length for field to fall by 10 %. 0 for hard edge or
            > 0 for soft edge with B0 / (1 + (z / d)**2)**2 dependence.
        :param field_extent: Extent of field beyond the main part. Given in
            units of edge_length.
        """
        self.center_pos = center_pos
        self.length = length
        self.B0 = B0
        self.direction = direction
        self.edge_scaled = edge_length / (10**0.5 - 1)**0.5
        self.field_extent = field_extent * edge_length

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
        print(b / (1 + (z / self.edge_scaled)**2.)**2.)
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
        zr = torch.abs(local_pos[2]) - 0.5 * self.length
        if zr < 0:
            return self.B0
        else:
            print(zr)
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

        # Create a boundary array
        self.bounds = []




    def get_field(self, position):
        pass


if __name__ == "__main__":
    test = Quadrupole(torch.tensor([0,0,0]), 1, 1, None, 0.5, 5)
    z = torch.linspace(-2, 2, 1000)
    x = torch.zeros_like(z) + 0.1
    p = torch.stack([x,x,z]).T

    f = torch.zeros(1000)
    for i in range(1000):
        f[i] = test.get_field(p[i])[1]

    fig, ax = plt.subplots()
    ax.plot(z, f)
    plt.show()