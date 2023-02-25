import torch
from .FieldSolver import FieldSolver
import matplotlib.pyplot as plt

# TODO need to add multi device looping


class BeamSolver:
    """
    Class for calculating the intensity of a beam of particles from coherent or
    incoherent radiation
    """

    def __init__(self, wavefront, track, optics_container=None, devices=None,
                 dt_args=None):
        """
        :param wavefront: Instance of Wavefront class
        :param track: Instance of Track class (should already have calculated
         single and bunch tracks
        :param optics_container: Instance of OpticsContainer class.
        :param devices: List of devices, e.g.
        :param dt_args: Dictionary of argument for automatic setting of time
         samples. If none then
        """

        if track.r is None or track.bunch_r is None:
            raise Exception("Track central / bunch trajectories have not been"
                            " set. Please run  sim_single_c and sim_bunch_c"
                            " before solving field")
        self.track = track
        self.wavefront = wavefront
        self.optics_container = optics_container
        self.devices = devices

        self.solver = FieldSolver(wavefront, track, device=devices)
        self.solver.set_dt(**dt_args, set_bunch=True)

    def solve_incoherent(self, n_part=None, blocks=1, solve_ends=True):
        """
        Calculates the intensity of incoherent radiation from multiple electrons
        :param n_part: Number of particles to simulate. Defaults to the
        number of tracks simulated. Also, can't be larger than number of
        available tracks
        :param blocks: Number of blocks to split calculation. Increasing this
         will reduce memory but slow calculation
         :param solve_ends: If true the integration is extended to +/- inf using
         an asymptotic expansion.
        :return: The intensity as a real 2d torch array
        """

        if n_part > self.track.bunch_r.shape[0]:
            raise Exception("Number of particles must be less than or equal "
                            "to the number of tracks simulated.")
        if not n_part:
            n_part = self.track.bunch_r.shape[0]

        intensity = torch.zeros_like(self.wavefront.x_array,
                                     device=self.devices, dtype=torch.double)

        for index in range(n_part):
            wavefront = self.solver.solve(blocks, solve_ends, True, index)
            if self.optics_container:
                self.optics_container.propagate(wavefront)
            intensity = intensity + wavefront.get_intensity()

        return intensity.T / n_part

    def solve_coherent(self, n_part):
        pass
