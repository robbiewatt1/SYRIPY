import numpy as np
import matplotlib.pyplot as plt
from scipy import interpolate
import cProfile
from Track import Track
from Wavefront import Wavefront

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
        # Loop over axis
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
                        self.track.time, phase_samples)
                    phase_func_der = phase_func.derivative()

                    # Form integrand function (f / gp)
                    n_dir_samples = r_obs[index, :] / r_norm
                    int1_samples = (self.track.beta[index] - n_dir_samples) \
                        / (r_norm * phase_func_der(self.track.time))
                    int2_samples = c_light * n_dir_samples \
                        / (self.wavefront.omega * r_norm**2.0
                           * phase_func_der(self.track.time))

                    int1_func = interpolate.InterpolatedUnivariateSpline(
                        phase_samples, int1_samples)
                    int2_func = interpolate.InterpolatedUnivariateSpline(
                        phase_samples, int2_samples)

                    phase_int = phase_func(t_int_points)

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
        oscillating sine function.
        :param func: Function to be integrated.
        :param omega: Oscillation frequency.
        :param x_samples: Integration sample points
        :return:
        """
        result = 0
        # loop samples
        for i in range(1, len(x_samples)):
            delta_x = (x_samples[i] - x_samples[i - 1]) / 2
            x1 = x_samples[i - 1]
            x2 = x_samples[i - 1] + delta_x
            x3 = x_samples[i]
            f1 = func(x1)
            f2 = func(x2)
            f3 = func(x3)

            # solve linear system
            f = np.array([f1, f2, f3]).T
            A = np.array([[1, 1, 1],
                          [x1, x2, x3],
                          [x1 ** 2.0, x2 ** 2.0, x3 ** 2.0]]).T

            c = np.linalg.solve(A, f)
            # Integrate window
            result += c[0] * self.sin_moment(omega, x1, x3) \
                      + c[1] * self.x_sin_moment(omega, x1, x3) \
                      + c[2] * self.x2_sin_moment(omega, x1, x3)
        return result

    def filon_cos(self, func, omega, x_samples):
        """
        :param func:
        :param omega:
        :param x_samples:
        :return: interal
        """
        result = 0
        # loop samples
        for i in range(1, len(x_samples)):
            delta_x = (x_samples[i] - x_samples[i - 1]) / 2
            x1 = x_samples[i - 1]
            x2 = x_samples[i - 1] + delta_x
            x3 = x_samples[i]
            f1 = func(x1)
            f2 = func(x2)
            f3 = func(x3)

            # solve linear system
            f = np.array([f1, f2, f3]).T
            A = np.array([[1, 1, 1],
                          [x1, x2, x3],
                          [x1 ** 2.0, x2 ** 2.0, 3 ** 2.0]]).T
            c = np.linalg.solve(A, f)
            result += c[0] * self.cos_moment(omega, x1, x3) \
                      + c[1] * self.x_cos_moment(omega, x1, x3) \
                      + c[2] * self.x2_cos_moment(omega, x1, x3)
        return result

    # Define the moments for the integration

    @staticmethod
    def sin_moment(omega, xi, xf):
        return (np.cos(omega * xi) - np.cos(omega * xf)) / omega

    @staticmethod
    def x_sin_moment(omega, xi, xf):
        return (np.sin(omega * xf) - xf * omega * np.cos(omega * xf)
                - np.sin(omega * xi) + xi * omega * np.cos(omega * xi)) \
               / omega**2

    @staticmethod
    def x2_sin_moment(omega, xi, xf):
        return ((2 - xf**2 * omega**2) * np.cos(omega * xf)
                + 2 * omega * xf * np.sin(omega * xf)
                - (2 - xi**2 * omega**2) * np.cos(omega * xi)
                - 2 * omega * xi * np.sin(omega * xi)) / omega**3.0

    @staticmethod
    def cos_moment(omega, xi, xf):
        return (np.sin(omega * xf) - np.sin(omega * xi)) / omega

    @staticmethod
    def x_cos_moment(omega, xi, xf):
        return (np.cos(omega * xf) + xf * omega * np.sin(omega * xf)
                - np.cos(omega * xi) - xi * omega * np.sin(omega * xi)) \
               / omega**2

    @staticmethod
    def x2_cos_moment(omega, xi, xf):
        return ((xf**2 * omega**2 - 2) * np.sin(omega * xf)
                + 2 * omega * xf * np.cos(omega * xf)
                - (xi**2 * omega**2 - 2) * np.sin(omega * xi)
                - 2 * omega * xi * np.cos(omega * xi)) / omega**3.0


if __name__ == "__main__":
    track = Track("./track.npy")
    track.plot_track(0, 9, 1000, [2, 0])
    plt.show()
    wavefnt = Wavefront(1.7526625849289021, 3.77e6,
                        np.linspace(-0.005, 0.005, 200),
                        np.linspace(-0.005, 0.005, 200))
    slvr = EdgeRadSolver(wavefnt, track)
    print(track.time[0], track.time[-1])
    cProfile.run("slvr.solve(0, 9, 1000)")
    wavefnt.plot_intensity()
    plt.show()

