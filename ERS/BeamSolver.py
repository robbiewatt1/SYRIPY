import torch
from .FieldSolver import FieldSolver
from .Wavefront import Wavefront
from .Tracking import Track
from .Optics import OpticsContainer
from typing import Optional, Dict, Tuple
import copy
import matplotlib.pyplot as plt

# TODO need to add multi device looping


class BeamSolver:
    """
    Class for calculating the intensity of a beam of particles from coherent or
    incoherent radiation
    """

    def __init__(self, wavefront: Wavefront, track: Track,
                 optics_container: Optional[OpticsContainer] = None,
                 dt_args: Optional[Dict[str, int]] = None,
                 compile_solver: bool = False) -> None:
        """
        :param wavefront: Instance of Wavefront class
        :param track: Instance of Track class (should already have calculated
         single and bunch tracks)
        :param optics_container: Instance of OpticsContainer class.
        :param dt_args: Dictionary of argument for automatic setting of time
         samples. If none default track is used.
        :param compile_solver: If true, the solver class will be compiled with
         torch.jit.script. This can speed the calculation up by ~ 2x.
        """

        if track.r is None or track.bunch_r is None:
            raise Exception("Track central / bunch trajectories have not been"
                            " set. Please run  sim_single_c and sim_bunch_c"
                            " before solving field")

        self.track = track
        self.wavefront = wavefront
        self.optics_container = optics_container
        self.solver = FieldSolver(wavefront, track)
        self.solver.set_dt(**dt_args, set_bunch=True)

        if compile_solver:
            self.solver = torch.jit.script(self.solver)

    def solve_incoherent(self, n_part: Optional[int] = None, blocks: int = 1,
                         solve_ends: bool = True, n_device: int = 1
                         ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Calculates the intensity of incoherent radiation from multiple electrons
        :param n_part: Number of particles to simulate. Defaults to the
         number of tracks simulated. Also, can't be larger than number of
         available tracks. If more than one device is used then the number of
         tracks simulated will be the largest whole number < n_part that is
         divisible by n_device.
        :param blocks: Number of blocks to split calculation. Increasing this
         will reduce memory but slow calculation
        :param solve_ends: If true the integration is extended to +/- inf using
         an asymptotic expansion.
        :param n_device: Number of devices to use. If > 1 then
        :return: The intensity as a real 2d torch array
        """
        # TODO if n_device > 1 check that its possible. I also assume track and
        #  initial solver are on cuda 0 so need to alo check that.
        #  If n_part = 1 make sure we use single device
        # Todo also return intensity array shape

        if n_part > self.track.bunch_r.shape[0]:
            raise Exception("Number of particles must be less than or equal "
                            "to the number of tracks simulated.")
        if not n_part:
            n_part = self.track.bunch_r.shape[0]

        # Get the field final x and y axes
        if (self.optics_container is None or self.optics_container.out_shape
                is None):
            x_axis = self.solver.wavefront.x_axis
            y_axis = self.solver.wavefront.y_axis
        else:
            x_axis = torch.linspace(self.optics_container.out_bounds[0],
                                    self.optics_container.out_bounds[1],
                                    self.optics_container.out_shape[0])
            y_axis = torch.linspace(self.optics_container.out_bounds[2],
                                    self.optics_container.out_bounds[3],
                                    self.optics_container.out_shape[1])

        if n_device == 1:  # Single device case, things are easy
            # Set the intensity array, depends on the final shape in the optics
            if (self.optics_container is None or self.optics_container.out_shape
                    is None):
                intensity_total = torch.zeros_like(self.wavefront.x_array,
                                                   device=self.solver.device)
            else:
                intensity_total = torch.zeros(self.optics_container.out_shape,
                                              device=self.solver.device)

            for index in range(n_part):  # Loop particles
                wavefront = self.solver.solve_field(blocks, solve_ends, True,
                                                    index)

                if self.optics_container is not None:
                    self.optics_container.propagate(wavefront)

                intensity_total += wavefront.get_intensity()

        else:  # multi device case is a bit harder
            # Make sure n particles is divisible by number of devices
            n_part = n_device * int(n_part / n_device)

            # Define the intensity tensors on all devices
            if (self.optics_container is None or self.optics_container.out_shape
                    is None):
                intensity_dev = [torch.zeros_like(
                    self.wavefront.x_array, device=torch.device(f"cuda:{i}"))
                    for i in range(n_device)]
            else:
                intensity_dev = [torch.zeros(self.optics_container.out_shape,
                                             device=torch.device(f"cuda:{i}"))
                                 for i in range(n_device)]

            # Set up the list of solvers and send to devices (confusing sorry!)
            solvers = [self.solver.solve_field]
            solvers.extend([copy.deepcopy(self.solver).switch_device(
                torch.device(f"cuda:{i}")).solve_field
                            for i in range(1, n_device)])

            for part_idx in range(0, n_part, n_device):  # Loop particles

                # Set the solver arguments
                arg_list = []
                for dev_idx in range(n_device):
                    args = [blocks, solve_ends, True, part_idx + dev_idx]
                    arg_list.append(args)

                # Solve the intensity and stack
                wavefronts = torch.nn.parallel.parallel_apply(solvers, arg_list)

                # Propagate wavefronts and sum
                for dev_idx in range(n_device):
                    if self.optics_container is not None:
                        self.optics_container.propagate(wavefronts[dev_idx])
                    intensity_dev[dev_idx] += wavefronts[dev_idx].get_intensity()

            # Now send everything to the first device which we assume is cuda:0
            intensity_total = torch.zeros_like(self.wavefront.x_array,
                                               device=torch.device("cuda:0"))
            for i, intensity in enumerate(intensity_dev):
                intensity_total += intensity.to("cuda:0")

        return intensity_total / n_part, x_axis, y_axis

    def solve_coherent(self, n_part):
        pass
