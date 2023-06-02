import torch
from .cTrack import cTrack
from typing import Optional, List


class FieldBlock(torch.nn.Module):
    """
    Base class of magnetic fields.
    """
    # TODO need to add the direction of the magnet
    def __init__(self, center_pos: torch.Tensor, length: float,
                 B0: torch.Tensor, direction: Optional[torch.Tensor] = None,
                 edge_length: float = 0., device: Optional[torch.device] = None
                 ) -> None:
        """
        :param center_pos: Central position of the magnet.
        :param length: Length of main part of field.
        :param B0: Field parameter vector [Units are tesla and meters.]
        :param direction: Vector through central axis of magnet [0, 0, 1]
            by default
        :param edge_length: Length for field to fall by 10 %. 0 for hard edge or
            > 0 for soft edge with B0 / (1 + (z / d)**2)**2 dependence.
        :param device: Device being used (e.g. cpu / gpu)
        """
        super().__init__()

        me = 9.1093837e-31
        qe = 1.60217663e-19
        self.B0 = (B0 / (me / (qe * 1.e-9))).to(device)

        self.center_pos = center_pos.to(device)
        self.length = length
        self.direction = direction  # Not yet implemented
        self.edge_length = edge_length
        self.edge_scaled = edge_length / (10.**0.5 - 1)**0.5
        self.order = 0

    def get_field(self, position: torch.Tensor) -> torch.Tensor:
        """
        Gets the field at a given location.
        :param position: Position of particle
        :return: Magnetic field vector.
        """
        return self.B0

    def _fridge(self, b: torch.Tensor, z: torch.Tensor) -> torch.Tensor:
        """
        Calculates the fringe field at a given location.
        :param b: Field vector at edge.
        :param z: Distance from end of main field.
        :return: Fringe field vector [bx, by, bz] (bx always 0)
        """
        return b[None] / (1 + (z[:, None] / self.edge_scaled)**2.)**2.


class Dipole(FieldBlock):
    """
    Defines a dipole field. The field is a constant value inside the main length
    and decays as
    """

    def __init__(self, center_pos: torch.Tensor, length: float,
                 B0: torch.Tensor, direction: Optional[torch.Tensor] = None,
                 edge_length: float = 0., device: Optional[torch.device] = None
                 ) -> None:
        """
        :param center_pos: Central position of the magnet.
        :param length: Length of main part of field.
        :param B0: Field strength of dipole.
        :param direction: Vector through central axis of magnet [0, 0, 1]
         by default
        :param edge_length: Length for field to fall by 10 %. 0 for hard edge or
         > 0 for soft edge with B0 / (1 + (z / d)**2)**2 dependence.
        :param device: Device being used (e.g. cpu / gpu)
        """
        super().__init__(center_pos, length, B0, direction, edge_length, device)
        self.order = 1

    def get_field(self, position: torch.Tensor) -> torch.Tensor:
        """
        Gets the field at a given location.
        :param position: Position of particle
        :return: Magnetic field vector.
        """
        local_pos = position - self.center_pos
        zr = torch.atleast_1d(torch.abs(local_pos[..., 2]) - 0.5 * self.length)
        inside = torch.where(zr < 0, 1, 0)
        outside = torch.abs(inside - 1)
        return torch.squeeze(inside[:, None] * self.B0 + outside[:, None]
                             * self._fridge(self.B0, zr))


class Quadrupole(FieldBlock):
    """
    Defines a Quadrupole field. The field increases linearly off axis. Positive
    gradient means focusing in x and negative is focusing in y.
    """

    def __init__(self, center_pos: torch.Tensor, length: float,
                 gradB: torch.Tensor, direction: Optional[torch.Tensor] = None,
                 edge_length: float = 0., device: Optional[torch.device] = None
                 ) -> None:
        """
        :param center_pos: Central position of the magnet.
        :param length: Length of main part of field.
        :param gradB: Gradient of field strength.
        :param direction: Vector through central axis of magnet [0, 0, 1]
            by default.
        :param edge_length: Length for field to fall by 10 %. 0 for hard edge or
            > 0 for soft edge with B0 / (1 + (z / d)**2)**2 dependence.
        :param device: Device being used (e.g. cpu / gpu)
        """
        super().__init__(center_pos, length, gradB, direction, edge_length,
                         device)

    def get_field(self, position: torch.Tensor) -> torch.Tensor:
        """
        Gets the field at a given location.
        :param position: Position of particle
        :return: Magnetic field vector.
        """
        local_pos = position - self.center_pos
        zr = torch.atleast_1d(torch.abs(local_pos[..., 2]) - 0.5 * self.length)
        inside = torch.where(zr < 0, 1, 0)
        outside = torch.abs(inside - 1)
        return torch.squeeze(
            inside[:, None] * self.B0 * local_pos[[1, 0, 2]] + outside[:, None]
            * self._fridge(self.B0 * local_pos[[1, 0, 2]], zr))


class FieldContainer(torch.nn.Module):
    """
    class containing a list of all defined magnetic elements. Will return the
    field for any given location.
    """

    def __init__(self, field_array: List[FieldBlock]) -> None:
        """
        :param field_array: A list containing all the defined elements.
        """
        super().__init__()
        self.field_array = torch.nn.ModuleList(field_array)

    def get_field(self, position: torch.Tensor) -> torch.Tensor:
        """
        Finds which element we are in and returns field
        :param position: Position of particle
        :return: Magnetic field vector.
        """
        field = torch.zeros_like(position)
        for element in self.field_array:
            field += element.get_field(position)
        return field

    @torch.jit.unused
    def gen_c_container(self) -> cTrack.FieldContainer:
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
