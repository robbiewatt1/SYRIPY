import numpy as np
import matplotlib.pyplot as plt
from scipy import interpolate
import cProfile
from Track import Track
from Wavefront import Wavefront
import time

c_light = 0.29979245


class EdgeRadSolver:
    """
    Class which solves the Liénard–Wiechert field at a wavefront for a given
    particle trajectory.
    """

    def __init__(self, wavefront, track):
        """
        :param wavefront: Instance of Wavefront class
        :param track: Instance of Track class
        """
        self.wavefront = wavefront
        self.track = track

    def auto_res(self, sample_x=10, ds_windows=100, ds_min=3):
        """
        Automatically down-samples the track based on large values of
        objective function obj = |grad(1/grad(g))|. Where g(t) is the phase
        function.
        :param sample_x: Number of test sample points in x
        :param ds_windows: Number of windows in which the path is down sampled
        :param ds_min: log10 of minimum down sampling factor
        """

        init_samples = self.track.time.shape[0]

        # Calculate grad(1/ grad(phi)) at sample_x points.
        test_index = int(self.wavefront.x_axis.shape[0] / sample_x)
        r_obs = np.stack([self.wavefront.x_axis[::test_index],
                          np.zeros(sample_x),
                          np.ones(sample_x) * self.wavefront.z]).T[:, :, None] \
                - self.track.r
        r_norm = np.linalg.norm(r_obs, axis=1)
        phase_samples = self.track.time[None, :] + r_norm / c_light
        phase_grad = np.gradient(phase_samples, self.track.time, axis=1)
        grad_inv_grad = np.gradient(1 / phase_grad, self.track.time, axis=1)
        obj_val = np.max(np.abs(grad_inv_grad), axis=0)

        # Calculate mean objective value in windows
        window_shape = (-1, int(obj_val.shape[0] / ds_windows))
        obj_val_windows = np.mean(obj_val.reshape(window_shape), axis=1)

        # Calculate the log10 down-sampling factor for each window
        window_ds_fact = -np.log10(obj_val_windows
                                   / np.max(obj_val_windows))
        # Clip down sampling at min value
        window_ds_fact = np.clip(window_ds_fact, a_min=0, a_max=ds_min)
        window_ds_fact = window_ds_fact

        # Apply down-sampling to track values
        # time
        time_list = list(self.track.time.reshape(window_shape))
        time_list = [t[::int(10**window_ds_fact[i])]
                     for i, t in enumerate(time_list)]
        self.track.time = np.concatenate(time_list)

        # Beta
        b_list = list(self.track.beta.reshape((3, *window_shape))
                      .transpose(1, 2, 0))
        beta_list = [b[::int(10**window_ds_fact[i])]
                     for i, b in enumerate(b_list)]
        self.track.beta = np.concatenate(beta_list).T

        # Position
        r_list = list(self.track.r.reshape((3, *window_shape))
                      .transpose(1, 2, 0))
        r_list = [r[::int(10**window_ds_fact[i])]
                  for i, r in enumerate(r_list)]
        self.track.r = np.concatenate(r_list).T
        print("Reduction factor: ", init_samples / self.track.time.shape[0])


    def solve(self, t_start, t_end, n_int, auto_res=True):
        """
        Main function to solve the radiation field at the wavefront.
        Interaction limits must be within time array of track
        :param t_start: Stat time of interation (ns)
        :param t_end: End time of
        :param n_int: Number of sample in integration
        :param auto_res: Automatically reduce resolution.
        """
        # check that t_start and t_end are acceptable
        f_interp = interpolate.interp1d(self.track.time, self.track.r[2])
        if t_start < self.track.time[0] or t_start > self.track.time[-1]:
            raise Exception(f"Integration boundaries must be within provided "
                            f"track. Track boundaries are"
                            f" {self.track.time[0]}  and {self.track.time[-1]}")
        if f_interp(t_end) > self.wavefront.z:
            raise Exception(f"Particle is beyond wavefront z at t_end."
                            f" z pos at time {t_end} is {f_interp(t_end)}")

        # First solve edge parts (still to do)

        # Set integration points
        start_index = np.argmin(np.abs(self.track.time - t_start))
        end_index = np.argmin(np.abs(self.track.time - t_end))
        skip = int((end_index - start_index) / n_int)+1
        t_int_points = self.track.time[start_index:end_index:skip]

        for index in [0, 1]:
            # loop through wavefront array
            for i, xi in enumerate(self.wavefront.x_axis):
                print(i)
                for j, yj in enumerate(self.wavefront.y_axis):
                    # Form phase function and phase function inverse
                    r_obs = np.array([xi, yj, self.wavefront.z])[:, None] \
                            - self.track.r

                    r_norm = np.linalg.norm(r_obs, axis=0)

                    # Calc phase / gradient
                    phase_samples = self.track.time + r_norm / c_light
                    phase_samples = phase_samples - phase_samples[0]
                    phase_grad = np.gradient(phase_samples, self.track.time)

                    # Phase function
                    phase_func = interpolate.InterpolatedUnivariateSpline(
                        self.track.time, phase_samples, k=3)

                    # Form integrand function (f / gp)
                    n_dir_samples = r_obs[index, :] / r_norm
                    int1_samples = (self.track.beta[index] - n_dir_samples) \
                        / (r_norm * phase_grad)
                    int2_samples = c_light * n_dir_samples \
                        / (self.wavefront.omega * r_norm**2.0
                           * phase_grad)

                    int1_func = interpolate.InterpolatedUnivariateSpline(
                        phase_samples, int1_samples, k=3)
                    int2_func = interpolate.InterpolatedUnivariateSpline(
                        phase_samples, int2_samples, k=3)



                    phase_int = phase_func(t_int_points)
                    """
                    fig, ax = plt.subplots()
                    ax.scatter(t_int_points, phase_int)
                    fig, ax = plt.subplots()
                    ax.scatter(t_int_points, int1_func(phase_int))
                    fig, ax = plt.subplots()
                    ax.scatter(t_int_points, int2_func(phase_int))
                    plt.show()
                    """
                    # Perform real and imaginary integrals
                    real_part = self.filon_cos(int1_func, self.wavefront.omega,
                                               phase_int) + \
                                self.filon_sin(int2_func, self.wavefront.omega,
                                               phase_int)
                    imag_part = self.filon_sin(int1_func, self.wavefront.omega,
                                               phase_int) - \
                                self.filon_cos(int2_func, self.wavefront.omega,
                                               phase_int)

                    self.wavefront.field[index, i, j] = real_part \
                                                        + 1j * imag_part

    def filon_sin(self, func, omega, x_samples):
        """
        Filon based method for integrating function multiplied by a rapidly
        oscillating sine wave. I = int[f(x) sin(omega x)], omega >> 1. Uses a
        quadratic approximation for f(x), allowing I to be solved analytically.
        :param func: Function to be integrated, f(x).
        :param omega: Oscillation frequency.
        :param x_samples: Sample points of integration.
        :return: Integration result.
        """

        # Create sample point array
        lower_lim = x_samples[:-1]
        upper_lim = x_samples[1:]
        mid_point = lower_lim + (upper_lim - lower_lim) / 2
        int_points = np.stack([lower_lim, mid_point, upper_lim]).T

        # Find coefficients of quadratic approximation (points are shifted to
        # make A less singular)
        f = func(int_points)
        ini_points_shift = int_points - int_points[:, 0, None]
        A = np.stack([np.ones_like(ini_points_shift), ini_points_shift,
                      ini_points_shift ** 2.0]).transpose((1, 2, 0))
        coeffs = np.linalg.solve(A, f)

        # Calculate moments and sum
        moment_0th = self.sin_moment(omega, int_points[:, ::2])
        moment_1st = self.x_sin_moment(omega, int_points[:, ::2])
        moment_2nd = self.x2_sin_moment(omega, int_points[:, ::2])
        return np.sum(coeffs[:, 0] * moment_0th + coeffs[:, 1] * moment_1st
                      + coeffs[:, 2] * moment_2nd)

    def filon_cos(self, func, omega, x_samples):
        """
        Filon based method for integrating function multiplied by a rapidly
        oscillating cosine wave. I = int[f(x) cos(omega x)], omega >> 1. Uses a
        quadratic approximation for f(x), allowing I to be solved analytically.
        :param func: Function to be integrated, f(x).
        :param omega: Oscillation frequency.
        :param x_samples: Sample points of integration.
        :return: Integration result.
        """

        # Create sample point array
        lower_lim = x_samples[:-1]
        upper_lim = x_samples[1:]
        mid_point = lower_lim + (upper_lim - lower_lim) / 2
        int_points = np.stack([lower_lim, mid_point, upper_lim]).T

        # Find coefficients of quadratic approximation (points are shifted to
        # make A less singular)
        f = func(int_points)
        ini_points_shift = int_points - int_points[:, 0, None]
        A = np.stack([np.ones_like(ini_points_shift), ini_points_shift,
                      ini_points_shift**2.0]).transpose((1, 2, 0))
        coeffs = np.linalg.solve(A, f)

        # Calculate moments and sum
        moment_0th = self.cos_moment(omega, int_points[:, ::2])
        moment_1st = self.x_cos_moment(omega, int_points[:, ::2])
        moment_2nd = self.x2_cos_moment(omega, int_points[:, ::2])
        return np.sum(coeffs[:, 0] * moment_0th + coeffs[:, 1] * moment_1st
                      + coeffs[:, 2] * moment_2nd)

    # Define the moments for the integration

    @staticmethod
    def sin_moment(omega, x_lim):
        """
        :param omega: Oscillation frequency
        :param x_lim: Integration limits [xi, xf]
        :return: Zeroth moment [sin(omega x)]
        """
        return (np.cos(omega * x_lim[..., 0])
                - np.cos(omega * x_lim[..., 1])) / omega

    @staticmethod
    def x_sin_moment(omega, x_lim):
        """
        :param omega: Oscillation frequency
        :param x_lim: Integration limits [xi, xf]
        :return: First moment [x sin(omega x)]
        """
        return (np.sin(omega * x_lim[..., 1]) - x_lim[..., 1] * omega
                * np.cos(omega * x_lim[..., 1]) - np.sin(omega * x_lim[..., 0])
                + x_lim[..., 0] * omega * np.cos(omega * x_lim[..., 0])) \
               / omega**2

    @staticmethod
    def x2_sin_moment(omega, x_lim):
        """
        :param omega: Oscillation frequency
        :param x_lim: Integration limits [xi, xf]
        :return: Second moment [x^2 sin(omega x)]
        """
        return ((2 - x_lim[..., 1]**2 * omega**2)
                * np.cos(omega * x_lim[..., 1]) + 2 * omega * x_lim[..., 1]
                * np.sin(omega * x_lim[..., 1])
                - (2 - x_lim[..., 0]**2 * omega**2)
                * np.cos(omega * x_lim[..., 0]) - 2 * omega * x_lim[..., 0]
                * np.sin(omega * x_lim[..., 0])) / omega**3.0

    @staticmethod
    def cos_moment(omega, x_lim):
        """
        :param omega: Oscillation frequency
        :param x_lim: Integration limits [xi, xf]
        :return: Zeroth moment [cos(omega x)]
        """
        return (np.sin(omega * x_lim[..., 1])
                - np.sin(omega * x_lim[..., 0])) / omega

    @staticmethod
    def x_cos_moment(omega, x_lim):
        """
        :param omega: Oscillation frequency
        :param x_lim: Integration limits [xi, xf]
        :return: First moment [x cos(omega x)]
        """
        return (np.cos(omega * x_lim[..., 1]) + x_lim[..., 1] * omega
                * np.sin(omega * x_lim[..., 1]) - np.cos(omega * x_lim[..., 0])
                - x_lim[..., 0] * omega * np.sin(omega * x_lim[..., 0])) \
               / omega**2

    @staticmethod
    def x2_cos_moment(omega, x_lim):
        """
        :param omega: Oscillation frequency
        :param x_lim: Integration limits [xi, xf]
        :return: Second moment [x^2 cos(omega x)]
        """
        return ((x_lim[..., 1]**2 * omega**2 - 2)
                * np.sin(omega * x_lim[..., 1]) + 2 * omega * x_lim[..., 1]
                * np.cos(omega * x_lim[..., 1])
                - (x_lim[..., 0]**2 * omega**2 - 2)
                * np.sin(omega * x_lim[..., 0]) - 2 * omega * x_lim[..., 0]
                * np.cos(omega * x_lim[..., 0])) / omega**3.0


if __name__ == "__main__":
    track = Track("./track.npy")
    track.plot_track(0, 9, 1000, [2, 0])
    plt.show()
    wavefnt = Wavefront(1.7526625849289021, 3.77e6,
                        np.linspace(-0.01, 0.01, 100),
                        np.linspace(-0.01, 0.01, 100))

    slvr = EdgeRadSolver(wavefnt, track)
    #slvr.auto_res(1)


    tic = time.perf_counter()
    slvr.solve(0, 9, 1000)
    toc = time.perf_counter()
    print(f"Downloaded the tutorial in {toc - tic:0.4f} seconds")
    wavefnt.plot_intensity()
    np.save("int", wavefnt.field)
    plt.show()

