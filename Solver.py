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

    def solve(self, t_start, t_end, n_int):
        """
        Main function to solve the radiation field at the wavefront.
        Interaction limits must be within time array of track
        :param t_start: Stat time of interation (ns)
        :param t_end: End time of
        :param n_int: Number of sample in integration
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

        # Integration sample points in time
        t_int_points = np.linspace(t_start, t_end, n_int)

        # define field
        #field = np.zeros((3, len(self.wavefront.x_axis),
        #                  len(self.wavefront.x_axis)), dtype=complex)

        # calculate observations
        #r_obs =

        for index in [0, 1]:
            # loop through wavefront array
            for i, xi in enumerate(self.wavefront.x_axis):
                print(i)
                for j, yj in enumerate(self.wavefront.y_axis):

                    # Form phase function and phase function inverse
                    r_obs = np.array([xi, yj, self.wavefront.z])[:, None] \
                            - self.track.r
                    r_norm = np.linalg.norm(r_obs, axis=0)
                    phase_samples = self.track.time + r_norm / c_light

                    # Shift phase samples for numerical stability
                    phase_samples = phase_samples - phase_samples[0]

                    # Phase function / derivative
                    phase_func = interpolate.InterpolatedUnivariateSpline(
                        self.track.time, phase_samples, k=2)
                    phase_func_der = phase_func.derivative()

                    # Form integrand function (f / gp)
                    n_dir_samples = r_obs[index, :] / r_norm
                    int1_samples = (self.track.beta[index] - n_dir_samples) \
                        / (r_norm * phase_func_der(self.track.time))
                    int2_samples = c_light * n_dir_samples \
                        / (self.wavefront.omega * r_norm**2.0
                           * phase_func_der(self.track.time))

                    int1_func = interpolate.InterpolatedUnivariateSpline(
                        phase_samples, int1_samples, k=2)
                    int2_func = interpolate.InterpolatedUnivariateSpline(
                        phase_samples, int2_samples, k=2)

                    phase_int = phase_func(t_int_points)

                    print(np.array([xi, yj, self.wavefront.z]))
                    fig, ax = plt.subplots()
                    ax.scatter(self.track.time, phase_samples)
                    ax.set_yscale("log")
                    fig, ax = plt.subplots()
                    ax.scatter(self.track.time, int1_samples, color="red")
                    ax.plot(t_int_points, int1_func(phase_int), color="blue")
                    fig, ax = plt.subplots()
                    ax.scatter(self.track.time, int2_samples, color="red")
                    ax.plot(t_int_points, int2_func(phase_int), color="blue")
                    fig, ax = plt.subplots()
                    ax.scatter(self.track.time, phase_func_der(self.track.time), color="red")
                    ax.plot(t_int_points, phase_func_der(t_int_points), color="blue")
                    ax.set_yscale("log")
                    plt.show()

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

        # Find coefficients of quadratic approximation
        f = func(int_points)
        A = np.stack([np.ones_like(int_points), int_points,
                      int_points ** 2.0]).transpose((1, 2, 0))
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

        # Find coefficients of quadratic approximation
        f = func(int_points)
        A = np.stack([np.ones_like(int_points), int_points,
                      int_points**2.0]).transpose((1, 2, 0))
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
                        np.linspace(-0.02, 0.02, 3),
                        np.linspace(-0.02, 0.02, 3))

    slvr = EdgeRadSolver(wavefnt, track)


    #func = lambda x : x**2.0
    #x_sample = np.linspace(0, 1, 50)
    #slvr.filon_sin(func, 1000, x_sample)
    #print(track.time[0], track.time[-1])
    tic = time.perf_counter()
    slvr.solve(0, 9, 50000)
    toc = time.perf_counter()
    print(f"Downloaded the tutorial in {toc - tic:0.4f} seconds")
    wavefnt.plot_intensity()
    np.save("int", wavefnt.field)
    plt.show()

