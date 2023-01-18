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

    def set_magnets(self):
        pass

    def sim_track(self):
        pass


