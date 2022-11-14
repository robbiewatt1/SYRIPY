import numpy as np
import torch
import matplotlib.pyplot as plt


class Track:
    """
    Class which handles the central electron beam track. Currently, this just
    loads a numpy array from an SRW simulation. I will add an RK4 solver soon.
    """

    def __init__(self, track_file, device=None):
        """
        :param track_file: SRW file containing particle track
        :param device: Device being used (cpu / gpu)
        """
        # Load from track file
        self.track = np.load(track_file)
        time = self.track[0]
        r = self.track[1:4]
        beta = self.track[4:]

        # Convert to pytorch
        self.time = torch.tensor(time, device=device)
        self.r = torch.tensor(r, device=device)
        self.beta = torch.tensor(beta, device=device)

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
