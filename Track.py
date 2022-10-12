import numpy as np
import matplotlib.pyplot as plt
from scipy import interpolate


class Track:
    """
    Class which handles the central electron beam track. Currently, this just
    loads a numpy array from an SRW simulation. I will add an RK4 solver soon.
    """

    def __init__(self, track_file):
        """
        :param track_file: SRW file containing particle track
        """
        self.track = np.load(track_file)
        self.time = self.track[0]
        self.r = self.track[1:4]
        self.beta = self.track[4:]

    def plot_track(self, t_start, t_end, n_samples, axes):
        """
        Plot interpolated track (Uses cubic spline)
        :param t_start: Start time (ns)
        :param t_end: End time(ns)
        :param n_samples: Sample points
        :param axes: Axes to plot (e. g z-x [2, 0])
        :return: fig, ax
        """
        time = np.linspace(t_start, t_end, n_samples)
        interp0 = interpolate.InterpolatedUnivariateSpline(self.time,
                                                           self.r[axes[0]])
        interp1 = interpolate.InterpolatedUnivariateSpline(self.time,
                                                           self.r[axes[1]])
        fig, ax = plt.subplots()
        ax.plot(interp0(time), interp1(time))
        return fig, ax
