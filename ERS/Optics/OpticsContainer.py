from ..Wavefront import Wavefront
from typing import List, Any
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
        self.out_bounds = None
        self.out_shape = None

        # Get the shape of the field at the end
        for element in element_array:
            if element.new_bounds:
                self.out_bounds = element.new_bounds
            if element.new_shape:
                self.out_shape = element.new_shape

    def add_element(self, element: OpticalElement) -> None:
        """
        :param element: Element to be added to the list.
        """
        self.element_array.append(element)

        # Check if new element updates output shape
        if element.new_bounds:
            self.out_bounds = element.new_bounds
        if element.new_shape:
            self.out_shape = element.new_shape

    def propagate(self, wavefront: Wavefront) -> None:
        """
        Propagates the wavefront through all the optical elements.
        :param wavefront: Wavefront to be propagated.
        """
        for element in self.element_array:
            element.propagate(wavefront)
