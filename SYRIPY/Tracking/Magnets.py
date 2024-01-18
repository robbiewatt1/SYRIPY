from .cTrack import cTrack
import torch
from typing import Optional, List
import math


class FieldBlock(torch.nn.Module):
    """
    Base class of magnetic fields.
    """
    # TODO need to add the direction of the magnet
    def __init__(self, center_pos: torch.Tensor, length: float,
                 B0: torch.Tensor, direction: Optional[torch.Tensor] = None,
                 edge_length: float = 0.) -> None:
        """
        :param center_pos: Central position of the magnet.
        :param length: Length of main part of field.
        :param B0: Field parameter vector [Units are tesla and meters.]
        :param direction: Vector through central axis of magnet [0, 0, 1]
         by default
        :param edge_length: Length for field to fall from 90% to 10 %.
         0 for hard edge or > 0 for soft edge with B0 / (1 + (z / d)**2)**2
         dependence.
        """
        super().__init__()

        me = 9.1093837e-31
        qe = 1.60217663e-19
        self.B0 = B0 / (me / (qe * 1.e-9))
        self.center_pos = center_pos
        self.length = length
        self.direction = direction  # Not yet implemented
        self.edge_length = edge_length
        self.edge_scaled = edge_length / 1.23789045853
        self.constant_length = 0.5 * (length - 1.2689299897 * edge_length)
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

    def switch_device(self, device: torch.device) -> None:
        """
        Switches the device of the field.
        :param device: Device to switch to.
        """
        self.B0 = self.B0.to(device)
        self.center_pos = self.center_pos.to(device)


class Dipole(FieldBlock):
    """
    Defines a dipole field. The field is a constant value inside the main length
    and decays as
    """

    def __init__(self, center_pos: torch.Tensor, length: float,
                 B0: torch.Tensor, direction: Optional[torch.Tensor] = None,
                 edge_length: float = 0.) -> None:
        """
        :param center_pos: Central position of the magnet.
        :param length: Length of main part of field.
        :param B0: Field strength of dipole.
        :param direction: Vector through central axis of magnet [0, 0, 1]
         by default
        :param edge_length: Length for field to fall by 10 %. 0 for hard edge or
         > 0 for soft edge with B0 / (1 + (z / d)**2)**2 dependence.
        """
        super().__init__(center_pos, length, B0, direction, edge_length)
        self.order = 1

    def get_field(self, position: torch.Tensor) -> torch.Tensor:
        """
        Gets the field at a given location.
        :param position: Position of particle
        :return: Magnetic field vector.
        """
        local_pos = position - self.center_pos
        zr = torch.atleast_1d(torch.abs(local_pos[..., 2])
                              - self.constant_length)
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
                 edge_length: float = 0.) -> None:
        """
        :param center_pos: Central position of the magnet.
        :param length: Length of main part of field.
        :param gradB: Gradient of field strength.
        :param direction: Vector through central axis of magnet [0, 0, 1]
            by default.
        :param edge_length: Length for field to fall by 10 %. 0 for hard edge or
            > 0 for soft edge with B0 / (1 + (z / d)**2)**2 dependence.
        """
        super().__init__(center_pos, length, gradB, direction, edge_length,
                         )

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

        # Make sure field array is sorted by position
        field_array.sort(key=lambda x: x.center_pos[2])
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

    def switch_device(self, device: torch.device) -> None:
        """
        Switches the device of the field.
        :param device: Device to switch to.
        """
        for element in self.field_array:
            element.switch_device(device)

    @torch.jit.ignore
    def get_transport_matrix(self, start_z: float, end_z: float,
                             gamma: torch.Tensor ) -> torch.Tensor:
        """
        Calculates the linear transport matrix for a given position. Only works
        for dispersion in x.
        :param start_z: Initial position of particle.
        :param end_z: Final position of particle.
        :param gamma: Beam design energy.
        :return: Both x and y beam matrix.
        """

        # TODO: I don't think this return differentiable tensors

        # Check element array is not empty
        if len(self.field_array) == 0:
            raise ValueError("No elements defined.")

        # Check we are not starting within an element
        for element in self.field_array:
            if (abs(start_z - element.center_pos[2]) < 0.5
                    * element.length):
                raise ValueError("Starting position is within a magnetic"
                                 " element.")

        # Find the first element
        next_element_index = 0
        for element in self.field_array:
            if start_z < element.center_pos[2]:
                break
            next_element_index += 1

        # Find the last element
        last_element_index = 0
        for element in self.field_array:
            if end_z < element.center_pos[2]:
                break
            last_element_index += 1

        current_z = start_z
        trans_matrix = torch.eye(6)
        for element in self.field_array[next_element_index:last_element_index]:
            # Propagate by drift to start of element
            drift_l = (element.center_pos[2] - 0.5 * element.length) - current_z
            trans_matrix = torch.matmul(self._drift_matrix(drift_l, gamma),
                                        trans_matrix)
            current_z += drift_l

            # Check if end point is within the next element
            element_l = element.length if current_z + element.length < end_z \
                else end_z - current_z
            current_z += element_l

            # Propagate through element
            if element.order == 1:
                trans_matrix = torch.matmul(
                    self._dipole_matrix(element.B0[1], element_l, gamma),
                    trans_matrix)
            elif element.order == 2:
                raise NotImplementedError("Quadrupole not yet implemented.")

        # Propagate by drift to end of element
        drift_l = end_z - current_z
        trans_matrix = torch.matmul(self._drift_matrix(drift_l, gamma),
                                    trans_matrix)
        current_z += drift_l

        return trans_matrix

    @staticmethod
    def _drift_matrix(length: float, gamma: float) -> torch.Tensor:
        """
        Calculates the drift matrix for a given length.
        :param length: Length of drift.
        :param gamma: Beam design energy.
        :return: Drift matrix.
        """
        beta = (1 - gamma**-2)**0.5
        return torch.tensor([[1., length, 0., 0., 0., 0.],
                             [0., 1., 0., 0., 0., 0.],
                             [0., 0., 1., length, 0., 0.],
                             [0., 0., 0., 1., 0., 0.],
                             [0., 0., 0., 0., 1., length / (beta * gamma)**2],
                             [0., 0., 0., 0., 0., 1.]])

    @staticmethod
    def _dipole_matrix(field: float, length: float, gamma: float
                       ) -> torch.Tensor:
        """
        Calculates the dipole matrix for a given field and length.
        :param field: Field strength.
        :param length: Length of dipole.
        :param gamma: Beam design energy.
        :return: Dipole matrix.
        """
        beta = (1 - gamma**-2)**0.5
        omega = field / (gamma * beta * 0.299792458)
        phi = omega * length
        trans_main = torch.tensor(
            [[math.cos(phi), math.sin(phi) / omega, 0., 0., 0.,
              (1. - math.cos(phi)) / (omega * beta)],
             [-omega * math.sin(phi), math.cos(phi), 0., 0., 0.,
              math.sin(phi) / beta],
             [0., 0., 1., length, 0., 0.],
             [0., 0., 0., 1., 0., 0.],
             [-math.sin(phi) / beta, - (1. - math.cos(phi)) / (omega * beta),
              0., 0., 1., length / (beta * gamma)**2 - (phi - math.sin(phi))
              / (omega * beta**2.)],
             [0., 0., 0., 0., 0., 1.]])
        k = -omega * math.tan(phi / 2.)
        trans_edge = torch.tensor([[1., 0., 0., 0., 0., 0.],
                                   [-k, 1., 0., 0., 0., 0.],
                                   [0., 0., 1., 0., 0., 0.],
                                   [0., 0., k, 1., 0., 0.],
                                   [0., 0., 0., 0., 1., 0.],
                                   [0., 0., 0., 0., 0., 1.]])
        return trans_edge @ trans_main @ trans_edge

    @staticmethod
    def _quadf_matrix(field: float, length: float, gamma: float
                      ) -> torch.Tensor:
        """
        Calculates the focusing quadrupole matrix for a given field and length.
        :param field: Field strength.
        :param length: Length of dipole.
        :param gamma: Beam design energy.
        :return: Quadrupole matrix.
        """
        beta = (1 - gamma**-2)**0.5
        omega = (field / (gamma * beta * 0.299792458))**0.5
        phi = omega * length
        return torch.tensor(
            [[math.cos(phi), math.sin(phi) / omega, 0., 0., 0., 0.],
                [-omega * math.sin(phi), math.cos(phi), 0., 0., 0., 0.],
                [0., 0., math.cosh(phi), math.sinh(phi) / omega, 0., 0.],
                [0., 0., omega * math.sinh(phi), math.cosh(phi), 0., 0.],
                [0., 0., 0., 0., 1., length / (beta * gamma)**2],
                [0., 0., 0., 0., 0., 1.]])

    @staticmethod
    def _quadd_matrix(field: float, length: float, gamma: float
                      ) -> torch.Tensor:
        """
        Calculates the defocusing quadrupole matrix for a given field and
        length.
        :param field: Field strength.
        :param length: Length of dipole.
        :param gamma: Beam design energy.
        :return: Quadrupole matrix.
        """
        beta = (1 - gamma**-2)**0.5
        omega = (-field / (gamma * beta * 0.299792458))**0.5
        phi = omega * length
        return torch.tensor(
            [[math.cosh(phi), math.sinh(phi) / omega, 0., 0., 0., 0.],
                [omega * math.sinh(phi), math.cosh(phi), 0., 0., 0., 0.],
                [0., 0., math.cos(phi), math.sin(phi) / omega, 0., 0.],
                [0., 0., -omega * math.sin(phi), math.cos(phi), 0., 0.],
                [0., 0., 0., 0., 1., length / (beta * gamma)**2],
                [0., 0., 0., 0., 0., 1.]])

    @torch.jit.unused
    def gen_c_container(self) -> cTrack.FieldContainer:
        """
        Converts the python field container class to the c-type
        :return: Instance of cTrack.FieldContainer with elements filled.
        """
        field_cont_c = cTrack.FieldContainer()
        for element in self.field_array:
            position = cTrack.ThreeVector(element.center_pos.tolist())
            field = cTrack.ThreeVector(element.B0.tolist())
            field_cont_c.addElement(element.order, position, field,
                                    element.length, element.edge_length)

        return field_cont_c

