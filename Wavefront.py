import numpy as np
import matplotlib.pyplot as plt


class Wavefront:
    """
    Wavefront class containing complex field array.
    """
    def __init__(self, z, omega, x_axis, y_axis):
        """
        :param z: Longitudinal position of wavefront
        :param omega: Frequency of radiation
        :param x_axis: Wavefront x_axis (e.g. np.linspace(...))
        :param y_axis: Wavefront x_axis (e.g. np.linspace(...))
        """
        self.z = z
        self.omega = omega
        self.x_axis = x_axis
        self.y_axis = y_axis
        self.x_array, self.y_array = np.meshgrid(x_axis, y_axis)
        #self.coords = np.stack([])
        self.field = np.zeros((3, len(x_axis), len(y_axis)), dtype=np.cdouble)

    def plot_intensity(self):
        """
        Plots the intensity of the wavefront.
        :return: (fig, ax)
        """
        intensity = (np.abs(self.field[0]) ** 2.0
                     + np.abs(self.field[1]) ** 2.0).T
        fig, ax = plt.subplots()
        ax.pcolormesh(intensity)
        return fig, ax
