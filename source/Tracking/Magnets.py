import torch
from cTrack import cTrack


me = 9.1093837e-31
qe = 1.60217663e-19


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
        self.edge_length = edge_length
        self.edge_scaled = edge_length / (10.**0.5 - 1)**0.5
        self.order = 0

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
        self.order = 1

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
                 edge_length=0.):
        """
        :param center_pos: Central position of the magnet.
        :param length: Length of main part of field.
        :param gradB: Gradient of field strength.
        :param direction: Vector through central axis of magnet [0, 0, 1]
            by defult
        :param edge_length: Length for field to fall by 10 %. 0 for hard edge or
            > 0 for soft edge with B0 / (1 + (z / d)**2)**2 dependence.
        """
        super().__init__(center_pos, length, torch.tensor([gradB, -gradB, 0]),
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

    def gen_c_container(self):
        """
        Converts the python field container class to the c-type
        :return: Instance of cTrack.FieldContainer with elements filled.
        """
        field_cont_c = cTrack.FieldContainer()
        for element in self.field_array:
            position = cTrack.ThreeVector(element.center_pos[0],
                                          element.center_pos[1],
                                          element.center_pos[2])
            field = cTrack.ThreeVector(element.B0[0],
                                       element.B0[1],
                                       element.B0[2])
            field_cont_c.addElement(element.order, position, field,
                                    element.length, element.edge_length)

        return field_cont_c
