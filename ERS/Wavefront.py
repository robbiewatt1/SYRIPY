import torch
import numpy as np
import matplotlib.pyplot as plt
from typing import Optional, List, Tuple


class Wavefront(torch.nn.Module):
    """
    Wavefront class containing complex field array.
    """

    def __init__(self, z: float, omega: float, wf_bounds: List[float],
                 n_samples_xy: List[int], dims: int = 2,
                 device: Optional[torch.device] = None) -> None:
        """
        :param z: Longitudinal position of wavefront
        :param omega: Frequency of radiation
        :param wf_bounds: Bounds of wavefront [x_min, x_max, y_min, y_max]
        :param n_samples_xy: Samples in wavefront [n_x, n_y]
        :param dims: Polarisation dimensions of field. (dims = 2 -> (x, y)
            3 -> (x, y, z))
        :param device: Device being used (e.g. cpu / gpu)
        """
        super().__init__()
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

        # Save the initial values in case we need to
        self.z_0 = z
        self.wf_bounds_0 = wf_bounds
        self.n_samples_xy_0 = n_samples_xy

    @torch.jit.export
    def pad_wavefront(self, pad_fact: int = 2) -> None:
        """
        Pads the field with zeros to prevent artifacts from fourier
        transform.
        :param pad_fact: Padding scaling factor
        """
        x_size_old, y_size_old = self.n_samples_xy[0], self.n_samples_xy[1]
        centre = [(self.wf_bounds[1] + self.wf_bounds[0]) / 2,
                  (self.wf_bounds[3] + self.wf_bounds[2]) / 2]
        length = [(self.wf_bounds[1] - self.wf_bounds[0]) / 2,
                  (self.wf_bounds[3] - self.wf_bounds[2]) / 2]
        self.wf_bounds = [centre[0] - pad_fact * length[0],
                          centre[0] + pad_fact * length[0],
                          centre[1] - pad_fact * length[1],
                          centre[1] + pad_fact * length[1]]
        self.n_samples_xy = [self.n_samples_xy[0] * pad_fact,
                             self.n_samples_xy[1] * pad_fact]
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

    def interpolate_wavefront(self, fact: int) -> None:
        """
        Increases the wavefront resolution using interpolation.
        :param fact: Factor to increase resolution by
        """
        self.field = self.field.reshape(2, self.n_samples_xy[0],
                                        self.n_samples_xy[1])
        # Stupid function doesn't work for complex so take this out as batch
        self.field = torch.stack([self.field.real, self.field.imag])
        self.field = torch.nn.functional.interpolate(
            self.field, scale_factor=(fact, fact), mode="bilinear",
            antialias=True)
        self.field = self.field[0] + 1j * self.field[1]
        self.field = self.field.flatten(1, 2)

        self.n_samples_xy = [self.n_samples_xy[0] * fact,
                             self.n_samples_xy[0] * fact]
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

    @torch.jit.export
    def update_bounds(self, bounds: List[float], n_samples_xy: List[int]
                      ) -> None:
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

    def change_dims(self, new_dims: int) -> None:
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

    def reset(self) -> None:
        """
        Resets the wavefront to the conditions when it was created. Used when
        calculating the bun ch intensity.
        """
        self.z = self.z_0
        self.wf_bounds = self.wf_bounds_0
        self.n_samples_xy = self.n_samples_xy_0
        self.n_samples = self.n_samples_xy[0] * self.n_samples_xy[1]
        self.delta = [
            (self.wf_bounds[1] - self.wf_bounds[0]) / self.n_samples_xy[0],
            (self.wf_bounds[3] - self.wf_bounds[2]) / self.n_samples_xy[1]]
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
        self.field = torch.zeros((self.dims, self.n_samples),
                                 dtype=torch.cfloat, device=self.device)

    def switch_device(self, device: torch.device) -> "Wavefront":
        """
        Changes the device that the class data is stored on.
        :param device: Device to switch to.
        """
        self.device = device
        self.x_axis = self.x_axis.to(device)
        self.y_axis = self.y_axis.to(device)
        self.x_array = self.x_array.to(device)
        self.coords = self.coords.to(device)
        self.field = self.field.to(device)
        return self

    @torch.jit.export
    def get_intensity(self) -> torch.Tensor:
        """
        Calculates and returns the intensity of the wavefront.
        :return: Intensity array
        """
        # Can't use abs with jit so need to do this with .real
        return torch.sum(self.field.real**2.0 + self.field.imag**2.0,
                         dim=0).reshape(self.n_samples_xy[0],
                                        self.n_samples_xy[1])

    def plot_intensity(self, log_plot: Optional[bool] = False,
                       ds_fact: int = 1, axes_lim: Optional[List[float]] = None,
                       lineout: Optional[List[int]] = None
                       ) -> Tuple[plt.Figure, plt.Axes]:
        """
        Plots the intensity of the wavefront.
        :param log_plot: Make intensity axis logged
        :param ds_fact: Down sample image to make plotting easier
        :param axes_lim: Sets the x/y axes limits [[xmin, xmax], [ymin, ymax]]
        :param lineout: Axis and index of lineout e.g [0, 50] will plot a
         lineout along y at x_i = 50. Default is None which plots 2d image
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
                pcol = ax.pcolormesh(
                    self.x_axis.cpu().detach().numpy()[::ds_fact],
                    self.y_axis.cpu().detach().numpy()[::ds_fact],
                    np.log10(intensity[::ds_fact, ::ds_fact]),
                    cmap="jet", shading="gouraud")
                fig.colorbar(pcol)
            else:
                pcol = ax.pcolormesh(
                    self.x_axis.cpu().detach().numpy()[::ds_fact],
                    self.y_axis.cpu().detach().numpy()[::ds_fact],
                    intensity[::ds_fact, ::ds_fact],
                    cmap="jet", shading="gouraud")
                fig.colorbar(pcol)

        if axes_lim:
            ax.set_xlim(axes_lim[0], axes_lim[1])
            ax.set_ylim(axes_lim[2], axes_lim[3])
        return fig, ax

    def plot_phase(self, dim: int = 0, ds_fact: int = 1,
                   axes_lim: Optional[List[float]] = None,
                   lineout: Optional[List[int]] = None
                   ) -> Tuple[plt.Figure, plt.Axes]:
        """
        Plots the intensity of the wavefront.
        :param dim: Dimension to plot.
        :param ds_fact: Down sample image to make plotting easier
        :param axes_lim: Sets the x/y axes limits [x_min, x_max, y_min, y_max]
        :param lineout: Axis and index of lineout e.g [0, 50] will plot a
         lineout along y at x_i = 50. Default is None which plots 2d image
        :return: (fig, ax)
        """
        phase = torch.angle(self.field[dim]).reshape(self.n_samples_xy[0],
                                                     self.n_samples_xy[1]).T

        fig, ax = plt.subplots()
        if lineout:  # 1D plot
            if lineout[0] == 0:
                ax.plot(self.y_axis.cpu().detach().numpy(),
                        phase.cpu().detach().numpy()[lineout[1], :])
            else:
                ax.plot(self.x_axis.cpu().detach().numpy(),
                        phase.cpu().detach().numpy()[:, lineout[1]])

        else:  # 2D plot
            pcol = ax.pcolormesh(
                self.x_axis.cpu().detach().numpy()[::ds_fact],
                self.y_axis.cpu().detach().numpy()[::ds_fact],
                phase.cpu().detach().numpy()[::ds_fact, ::ds_fact],
                cmap="jet", shading="gouraud")
            fig.colorbar(pcol)

        if axes_lim:
            ax.set_xlim(axes_lim[0], axes_lim[1])
            ax.set_ylim(axes_lim[2], axes_lim[3])
        return fig, ax
