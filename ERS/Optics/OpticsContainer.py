import torch
from ..Wavefront import Wavefront
from typing import List
from .OpticalElement import OpticalElement


class OpticsContainer:
    """
    Container of optical elements. Used to define a beamline and propagate
    through all elements.
    """

    def __init__(self, element_array: List[OpticalElement]) -> None:
        """
        :param element_array: List of the optical elements. Propagation occurs
         in order.
        """
        self.element_array = element_array

    def add_element(self, element: OpticalElement) -> None:
        """
        :param element: Element to be added to the list.
        """
        self.element_array.append(element)

    def propagate(self, wavefront: Wavefront) -> None:
        """
        Propagates the wavefront through all the optical elements.
        :param wavefront: Wavefront to be propagated.
        """
        for element in self.element_array:
            element.propagate(wavefront)

    def get_propagation_matrix(self) -> torch.Tensor:
        """
        Returns the linear propagation matrix for the element.
        :return: Propagation matrix.
        """
        prop_matrix = torch.eye(6)
        for element in self.element_array:
            prop_matrix = element.get_propagation_matrix() @ prop_matrix
        return prop_matrix
