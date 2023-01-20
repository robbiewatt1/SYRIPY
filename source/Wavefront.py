import torch
import numpy as np
import matplotlib.pyplot as plt


class Wavefront(torch.nn.Module):
    """
    Wavefront class containing complex field array.
    """
    def __init__(self, z, omega, wf_bounds, n_samples_xy, device=None):
        """
        :param z: Longitudinal position of wavefront
        :param omega: Frequency of radiation
        :param wf_bounds: Bounds of wavefront [xmin, xmax, ymin, ymax]
        :param n_samples_xy: Samples in wavefront [n_x, n_y]
        :param device: Device being used (e.g. cpu / gpu)
        """
        super().__init__()
        self.z = z
        self.omega = omega
        self.wf_bounds = wf_bounds
        self.n_samples_xy = n_samples_xy
        self.device = device
        self.n_samples = n_samples_xy[0] * n_samples_xy[1]
        self.delta = [(wf_bounds[1] - wf_bounds[0]) / n_samples_xy[0],
                      (wf_bounds[3] - wf_bounds[2]) / n_samples_xy[1]]
        self.x_axis = torch.linspace(wf_bounds[0], wf_bounds[1],
                                     n_samples_xy[0], device=device)
        self.y_axis = torch.linspace(wf_bounds[2], wf_bounds[3],
                                     n_samples_xy[1], device=device)
        self.x_array, self.y_array = torch.meshgrid(self.x_axis, self.y_axis,
                                                    indexing="ij")
        self.coords = torch.stack((self.x_array, self.y_array,
                                   z * torch.ones_like(self.x_array)),
                                  dim=2).flatten(0, 1)
        self.field = torch.zeros((self.n_samples, 2), dtype=torch.cfloat,
                                 device=device)
        print(self.coords.get_device(), self.y_array.get_device(), self.field.get_device())

    def pad_wavefront(self, pad_fact=2):
        """
        Pads the field with zeros to prevent artifacts from fourier
        transform.
        :param pad_fact: Padding scaling factor
        """
        x_size_old, y_size_old = self.n_samples_xy[0], self.n_samples_xy[1]
        self.wf_bounds = self.wf_bounds * pad_fact
        self.n_samples_xy = self.n_samples_xy * pad_fact
        self.n_samples = self.n_samples_xy[0] * self.n_samples_xy[1]
        self.x_axis = torch.linspace(self.wf_bounds[0], self.wf_bounds[1],
                                     self.n_samples_xy[0], device=self.device)
        self.y_axis = torch.linspace(self.wf_bounds[2], self.wf_bounds[3],
                                     self.n_samples_xy[1], device=self.device)
        self.x_array, self.y_array = torch.meshgrid(self.x_axis, self.y_axis,
                                                    indexing="ij")
        self.coords = torch.stack((self.x_array, self.y_array,
                                   self.z * torch.ones_like(self.x_array)),
                                  dim=2).flatten(0, 1)
        new_field = torch.zeros((self.n_samples_xy[0], self.n_samples_xy[1], 2),
                                dtype=torch.cfloat, device=self.device)
        new_field[self.n_samples_xy[0] // 2 - x_size_old // 2:
                  self.n_samples_xy[0] // 2 + x_size_old // 2,
                  self.n_samples_xy[1] // 2 - y_size_old // 2:
                  self.n_samples_xy[1] // 2 + y_size_old // 2, :] \
            = self.field.reshape(x_size_old, y_size_old, 2)
        self.field = new_field.flatten(0, 1)

    def update_bounds(self, bounds):
        """
        Updates the bounds of the wavefront and changes the grid coords.
        :param bounds: New bounds of wavefront [xmin, xmax, ymin, ymax]
        """
        self.wf_bounds = bounds
        self.x_axis = torch.linspace(self.wf_bounds[0], self.wf_bounds[1],
                                     self.n_samples_xy[0], device=self.device)
        self.y_axis = torch.linspace(self.wf_bounds[2], self.wf_bounds[3],
                                     self.n_samples_xy[1], device=self.device)
        self.x_array, self.y_array = torch.meshgrid(self.x_axis, self.y_axis,
                                                    indexing="ij")
        self.coords = torch.stack((self.x_array, self.y_array,
                                   self.z * torch.ones_like(self.x_array)),
                                  dim=2).flatten(0, 1)

    def plot_intensity(self, log_plot=False, axes_lim=None, ds_fact=1):
        """
        Plots the intensity of the wavefront.
        :return: (fig, ax)
        """
        intensity = (torch.abs(self.field[:, 0])**2.0
                     + torch.abs(self.field[:, 1])**2.0).cpu().detach().numpy()
        intensity = intensity.reshape(self.n_samples_xy[0],
                                      self.n_samples_xy[1]).T
        fig, ax = plt.subplots()

        if log_plot:
            pcol = ax.pcolormesh(self.x_axis.cpu().detach().numpy()[::ds_fact],
                                 self.y_axis.cpu().detach().numpy()[::ds_fact],
                                 np.log10(intensity[::ds_fact, ::ds_fact]),
                                 cmap="jet")
            fig.colorbar(pcol)
        else:
            pcol = ax.pcolormesh(self.x_axis.cpu().detach().numpy()[::ds_fact],
                                 self.y_axis.cpu().detach().numpy()[::ds_fact],
                                 intensity[::ds_fact, ::ds_fact],
                                 cmap="jet")
            fig.colorbar(pcol)

        if axes_lim:
            ax.set_xlim(axes_lim[0], axes_lim[1])
            ax.set_ylim(axes_lim[2], axes_lim[3])
        return fig, ax
