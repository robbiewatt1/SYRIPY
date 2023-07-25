import torch


class BilinearInterp:
    """
    A class to perform bilinear interpolation on a 2D field. Can be used with
    batched field. Will return zeros when extrapolating.
    """

    def __init__(self, x_axis: torch.Tensor, y_axis: torch.Tensor,
                 field: torch.Tensor) -> None:
        """
        :param x_axis: 1D array of x-axis values
        :param y_axis: 1D array of y-axis values
        :param field: 2D array of field values
        """
        self.x_axis = x_axis
        self.y_axis = y_axis
        self.field = field

        # Get batch size
        if len(field.shape) == 2:
            self.batch_size = 1
        else:
            self.batch_size = field.shape[0]

    def __call__(self, new_x_axis: torch.Tensor, new_y_axis: torch.Tensor
                 ) -> torch.Tensor:
        """
        :param new_x_axis: 1D array of x-axis values to interpolate to
        :param new_y_axis: 1D array of y-axis values to interpolate to
        returns: 2D array of interpolated field values
        """

        # Get sample points
        x_samples, y_samples = torch.meshgrid(new_x_axis, new_y_axis,
                                              indexing='ij')

        x_samples = x_samples.flatten()
        y_samples = y_samples.flatten()

        # Get the indices of the points to interpolate to
        x_idx = torch.searchsorted(self.x_axis, x_samples)
        y_idx = torch.searchsorted(self.y_axis, y_samples)

        # Check if indices are within the range of the axis
        interp_bool = torch.logical_and(
            torch.logical_and(torch.gt(x_idx, 0),
                              torch.lt(x_idx, self.x_axis.shape[0])),
            torch.logical_and(torch.gt(y_idx, 0),
                              torch.lt(y_idx, self.y_axis.shape[0])))

        # Wrap indices back around
        x_low = (x_idx - 1) % self.x_axis.shape[0]
        x_high = x_idx % self.x_axis.shape[0]
        y_low = (y_idx - 1) % self.y_axis.shape[0]
        y_high = y_idx % self.y_axis.shape[0]

        # Get the x and y values of the points to interpolate to
        x0 = self.x_axis[x_low]
        x1 = self.x_axis[x_high]
        y0 = self.y_axis[y_low]
        y1 = self.y_axis[y_high]

        # Get the field values of the points to interpolate to
        f00 = torch.where(interp_bool, self.field[..., x_low, y_low], 0)
        f01 = torch.where(interp_bool,  self.field[..., x_low, y_high], 0)
        f10 = torch.where(interp_bool,  self.field[..., x_high, y_low], 0)
        f11 = torch.where(interp_bool, self.field[..., x_high, y_high], 0)

        # Calculate the interpolated field values
        f_interp = ((f00 * (x1 - x_samples) * (y1 - y_samples) +
                     f10 * (x_samples - x0) * (y1 - y_samples) +
                     f01 * (x1 - x_samples) * (y_samples - y0) +
                     f11 * (x_samples - x0) * (y_samples - y0)) /
                    ((x1 - x0) * (y1 - y0)))
        return torch.squeeze(f_interp.reshape(self.batch_size, len(new_x_axis),
                                              len(new_y_axis)))


class CubicInterp:
    """
    Cubic spline interpolator class using pytorch. Can perform multiple
    interpolations along batch dimension.
    """

    def __init__(self, x: torch.Tensor, y: torch.Tensor) -> None:
        """
        :param x: x samples
        :param y: y samples
        """
        self.x = x
        self.y = y
        self.device = x.device
        self.A = torch.tensor([
            [1, 0, -3, 2],
            [0, 1, -2, 1],
            [0, 0, 3, -2],
            [0, 0, -1, 1]],
            dtype=x.dtype, device=self.device)

    def h_poly(self, x: torch.Tensor) -> torch.Tensor:
        """
        Calculates the Hermite polynomials
        :param x: Locations to calculate polynomials at
        :return: Hermite polynomials
        """
        xx = x[..., None, :] ** torch.arange(
            4, device=self.device)[..., :, None]
        return torch.matmul(self.A, xx)

    def __call__(self, xs: torch.Tensor) -> torch.Tensor:
        """
        Performs interpolation at location xs.
        :param xs: locations to interpolate
        :return: Interpolated value
        """
        m = ((self.y[..., 1:] - self.y[..., :-1]) /
             (self.x[..., 1:] - self.x[..., :-1]))
        m = torch.cat([m[..., [0]], (m[..., 1:] + m[..., :-1]) / 2,
                       m[..., [-1]]], dim=-1)
        idx = torch.searchsorted(self.x[..., 1:].contiguous(), xs)
        dx = (torch.gather(self.x, dim=-1, index=idx+1)
              - torch.gather(self.x, dim=-1, index=idx))
        hh = self.h_poly((xs - torch.gather(self.x, dim=-1, index=idx)) / dx)
        return (hh[..., 0, :] * torch.gather(self.y, dim=-1, index=idx)
                + hh[..., 1, :] * torch.gather(m, dim=-1, index=idx)
                * dx + hh[..., 2, :] * torch.gather(self.y, dim=-1, index=idx+1)
                + hh[..., 3, :] * torch.gather(m, dim=-1, index=idx+1) * dx)
