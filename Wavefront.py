import torch
import matplotlib.pyplot as plt


class Wavefront:
    """
    Wavefront class containing complex field array.
    """
    def __init__(self, z, omega, wf_bounds, n_samples_xy, device=None):
        """
        :param z: Longitudinal position of wavefront
        :param omega: Frequency of radiation
        :param wf_bounds: Bounds of wavefront [xmin, xmax, ymin, ymax]
        :param n_samples_xy: SAmples in wavefron [n_x, n_y]
        :param device: Device being used (e.g. cpu / gpu)
        """
        self.z = z
        self.omega = omega
        self.wf_bounds = wf_bounds
        self.n_samples_xy = n_samples_xy
        self.n_samples = n_samples_xy[0] * n_samples_xy[1]
        self.x_axis = torch.linspace(wf_bounds[0], wf_bounds[1],
                                     n_samples_xy[0], device=device)
        self.y_axis = torch.linspace(wf_bounds[2], wf_bounds[3],
                                     n_samples_xy[1], device=device)
        self.x_array, self.y_array = torch.meshgrid(self.x_axis, self.y_axis,
                                                    indexing="xy")
        # Create coordinate array
        self.coords = torch.stack((self.x_array, self.y_array,
                        z * torch.ones_like(self.x_array)), dim=2).flatten(0, 1)
        self.field = torch.zeros((n_samples_xy[0], n_samples_xy[1], 3),
                                 dtype=torch.cfloat)

    def plot_intensity(self):
        """
        Plots the intensity of the wavefront.
        :return: (fig, ax)
        """
        intensity = (torch.abs(self.field[:, :, 0]) ** 2.0
                     + torch.abs(self.field[:, :, 1]) ** 2.0
                     ).cpu().detach().numpy()
        fig, ax = plt.subplots()
        ax.pcolormesh(intensity)
        return fig, ax
