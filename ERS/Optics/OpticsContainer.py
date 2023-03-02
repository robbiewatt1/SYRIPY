

class OpticsContainer:
    """
    Container of optical elements. Used to define a beamline and propagate
    through all elements.
    """

    def __init__(self, element_array):
        """
        :param element_array: List of the optical elements. Propagation occurs
         in order.
        """
        self.element_array = element_array

    def add_element(self, element):
        """
        :param element: Element to be added to the list.
        """
        self.element_array.append(element)

    def propagate(self, wavefront):
        """
        Propagates the wavefront through all the optical elements.
        :param wavefront: Wavefront to be propagated.
        """
        for element in self.element_array:
            element.propagate(wavefront)
