import torch
import numpy as np
import matplotlib.pyplot as plt


class Wavefront:
    """
    Wavefront class containing complex field array.
    """
    def __init__(self, z, omega, wf_bounds, n_samples_xy, dims=2,
                 device=None):
        """
        :param z: Longitudinal position of wavefront
        :param omega: Frequency of radiation
        :param wf_bounds: Bounds of wavefront [xmin, xmax, ymin, ymax]
        :param n_samples_xy: Samples in wavefront [n_x, n_y]
        :param dims: Polarisation dimensions of field.  (dims = 2 -> (x, y)
            3 -> (x, y, z))
        :param device: Device being used (e.g. cpu / gpu)
        """
        self.z = z
        self.omega = omega
        self.wf_bounds = wf_bounds
        self.n_samples_xy = n_samples_xy
        self.dims = dims
        self.device = device
        self.n_samples = n_samples_xy[0] * n_samples_xy[1]
        self.delta = [(wf_bounds[1] - wf_bounds[0]) / n_samples_xy[0],
                      (wf_bounds[3] - wf_bounds[2]) / n_samples_xy[1]]
        self.x_axis = torch.linspace(wf_bounds[0] + self.delta[0] / 2,
                                     wf_bounds[1], n_samples_xy[0],
                                     device=device)
        self.y_axis = torch.linspace(wf_bounds[2] + self.delta[1] / 2,
                                     wf_bounds[3], n_samples_xy[1],
                                     device=device)
        self.x_array, self.y_array = torch.meshgrid(self.x_axis, self.y_axis,
                                                    indexing="ij")
        self.coords = torch.stack((self.x_array, self.y_array,
                                   z * torch.ones_like(self.x_array)),
                                  dim=0).flatten(1, 2)
        self.field = torch.zeros((dims, self.n_samples), dtype=torch.cfloat,
                                 device=device)

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
        self.x_axis = torch.linspace(self.wf_bounds[0] + self.delta[0] / 2,
                                     self.wf_bounds[1], self.n_samples_xy[0],
                                     device=self.device)
        self.y_axis = torch.linspace(self.wf_bounds[2] + self.delta[1] / 2,
                                     self.wf_bounds[3], self.n_samples_xy[1],
                                     device=self.device)
        self.x_array, self.y_array = torch.meshgrid(self.x_axis, self.y_axis,
                                                    indexing="ij")
        self.coords = torch.stack((self.x_array, self.y_array,
                                   self.z * torch.ones_like(self.x_array)),
                                  dim=0).flatten(1, 2)
        new_field = torch.zeros((self.dims, self.n_samples_xy[0],
                                 self.n_samples_xy[1]), dtype=torch.cfloat,
                                device=self.device)
        new_field[:, self.n_samples_xy[0] // 2 - x_size_old // 2:
                  self.n_samples_xy[0] // 2 + x_size_old // 2,
                  self.n_samples_xy[1] // 2 - y_size_old // 2:
                  self.n_samples_xy[1] // 2 + y_size_old // 2] \
            = self.field.reshape(self.dims, x_size_old, y_size_old)
        self.field = new_field.flatten(1, 2)

    def update_bounds(self, bounds, n_samples_xy):
        """
        Updates the bounds of the wavefront and changes the grid coords.
        :param bounds: New bounds of wavefront [xmin, xmax, ymin, ymax]
        :param n_samples_xy: Samples in wavefront [n_x, n_y]
        """
        self.wf_bounds = bounds
        self.n_samples_xy = n_samples_xy
        self.n_samples = n_samples_xy[0] * n_samples_xy[1]
        self.x_axis = torch.linspace(self.wf_bounds[0] + self.delta[0] / 2,
                                     self.wf_bounds[1], self.n_samples_xy[0],
                                     device=self.device)
        self.y_axis = torch.linspace(self.wf_bounds[2] + self.delta[1] / 2,
                                     self.wf_bounds[3], self.n_samples_xy[1],
                                     device=self.device)
        self.x_array, self.y_array = torch.meshgrid(self.x_axis, self.y_axis,
                                                    indexing="ij")
        self.coords = torch.stack((self.x_array, self.y_array,
                                   self.z * torch.ones_like(self.x_array)),
                                  dim=0).flatten(1, 2)

    def change_dims(self, new_dims):
        """
        Changes dimensions of the wavefront. If decreasing dims then z / y-axis
         is removed in that order
        :param new_dims: New dimensions of the wavefront
        """

        new_field = torch.zeros((new_dims, self.n_samples),
                                dtype=torch.cfloat,
                                device=self.device).flatten(1, 2)
        if self.dims == 1:
            new_field[0, :] = self.field
        elif self.dims == 2:
            if new_dims == 1:
                new_field[0, :] = self.field[0, :]
            elif new_dims == 3:
                new_field[:2, :] = self.field
        elif self.dims == 3:
            if new_dims == 1:
                new_field[0, :] = self.field[0, :]
            elif new_dims == 2:
                new_field[:2, :] = self.field
        self.field = new_field
        self.dims = new_dims

    def copy(self):
        """
        Copies the wavefront structure to a new instance with zero field
        :return: A copy of the wavefront.
        """
        return Wavefront(self.z, self.omega, self.wf_bounds, self.n_samples_xy,
                         self.dims, self.device)

    def get_intensity(self):
        """
        Calculates and returns the intensity of the wavefront.
        :return: Intensity array
        """
        return torch.sum(torch.abs(self.field)**2.0, dim=0)\
            .reshape(self.n_samples_xy[0], self.n_samples_xy[1])

    def plot_intensity(self, log_plot=False, axes_lim=None, ds_fact=1,
                       lineout=None):
        """
        Plots the intensity of the wavefront.
        :param log_plot: Make intensity axis logged
        :param axes_lim: Sets the x/y axes limits [[xmin, xmax], [ymin, ymax]]
        :param ds_fact: Down sample image to make plotting easier
        :param lineout: Axis and index of lineout e.g [0, 50] will plot a
         lineout along y at x_i = 50. Defult is None which plots 2d image
        :return: (fig, ax)
        """
        if self.dims == 1:
            intensity = (torch.abs(self.field[0, :]) ** 2.0
                         ).cpu().detach().numpy()
        elif self.dims == 2:
            intensity = (torch.abs(self.field[0, :])**2.0 +
                         torch.abs(self.field[1, :])**2.0
                         ).cpu().detach().numpy()
        else:
            intensity = (torch.abs(self.field[0, :])**2.0 +
                         torch.abs(self.field[1, :])**2.0 +
                         torch.abs(self.field[2, :])**2.0
                         ).cpu().detach().numpy()
        intensity = intensity.reshape(self.n_samples_xy[0],
                                      self.n_samples_xy[1]).T
        fig, ax = plt.subplots()
        if lineout:  # 1D plot
            if lineout[0] == 0:
                ax.plot(intensity[lineout[1], :])
            else:
                ax.plot(intensity[:, lineout[1]])
        else:  # 2D plot
            if log_plot:
                pcol = ax.pcolormesh(self.x_axis.cpu().detach().numpy()[::ds_fact],
                                     self.y_axis.cpu().detach().numpy()[::ds_fact],
                                     np.log10(intensity[::ds_fact, ::ds_fact]),
                                     cmap="jet", shading='auto')
                fig.colorbar(pcol)
            else:
                pcol = ax.pcolormesh(self.x_axis.cpu().detach().numpy()[::ds_fact],
                                     self.y_axis.cpu().detach().numpy()[::ds_fact],
                                     intensity[::ds_fact, ::ds_fact],
                                     cmap="jet", shading='auto')
                fig.colorbar(pcol)

        if axes_lim:
            ax.set_xlim(axes_lim[0], axes_lim[1])
            ax.set_ylim(axes_lim[2], axes_lim[3])
        return fig, ax

    def plot_phase(self, dim=0, axes_lim=None, ds_fact=1, lineout=None):
        """
        Plots the intensity of the wavefront.
        :param axes_lim: Sets the x/y axes limits [[xmin, xmax], [ymin, ymax]]
        :param ds_fact: Down sample image to make plotting easier
        :param lineout: Axis and index of lineout e.g [0, 50] will plot a
         lineout along y at x_i = 50. Defult is None which plots 2d image
        :return: (fig, ax)
        """
        phase = torch.angle(self.field[dim]).reshape(self.n_samples_xy[0],
                                                     self.n_samples_xy[1]).T
        fig, ax = plt.subplots()
        ax.pcolormesh(phase.cpu().detach().numpy(),
                      cmap="jet", shading='auto')

        fig, ax = plt.subplots()
        if lineout:  # 1D plot
            if lineout[0] == 0:
                ax.plot(self.y_axis.cpu().detach().numpy(),
                        phase[lineout[1], :])
            else:
                ax.plot(self.x_axis.cpu().detach().numpy(),
                        phase[:, lineout[1]])

        else:  # 2D plot
            pcol = ax.pcolormesh(
                self.x_axis.cpu().detach().numpy()[::ds_fact],
                self.y_axis.cpu().detach().numpy()[::ds_fact],
                phase[::ds_fact, ::ds_fact],
                cmap="jet", shading='auto')
            fig.colorbar(pcol)

        if axes_lim:
            ax.set_xlim(axes_lim[0], axes_lim[1])
            ax.set_ylim(axes_lim[2], axes_lim[3])
        return fig, ax

