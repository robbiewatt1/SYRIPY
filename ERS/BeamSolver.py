import torch
from .FieldSolver import FieldSolver
from .Wavefront import Wavefront
from .Tracking import Track
from .Optics import OpticsContainer
from typing import Optional, Dict


class BeamSolver:
    """
    Class for calculating the intensity of a beam of particles from coherent or
    incoherent radiation
    """

    def __init__(self, wavefront: Wavefront, track: Track,
                 optics_container: Optional[OpticsContainer] = None,
                 dt_args: Optional[Dict[str, int]] = None,
                 compile_solver: bool = False, blocks: int = 1,
                 batch_solve: int = 1, solve_ends: bool = True) -> None:
        """
        :param wavefront: Instance of Wavefront class
        :param track: Instance of Track class (should already have calculated
         single and bunch tracks)
        :param optics_container: Instance of OpticsContainer class.
        :param dt_args: Dictionary of argument for automatic setting of time
         samples. If none default track is used.
        :param compile_solver: If true, the solver class will be compiled with
         torch.jit.script. This can speed the calculation up by ~ 2x.
        :param blocks: Number of blocks to split calculation. Increasing this
         will reduce memory but slow calculation
        :param batch_solve: If larger than 1 then the solver will be batched,
         improving speed but increasing memory.
        :param solve_ends: If true the integration is extended to +/- inf using
         an asymptotic expansion.
        """

        if track.r is None or track.bunch_r is None:
            raise Exception("Track central / bunch trajectories have not been"
                            " set. Please run  sim_single_c and sim_bunch_c"
                            " before solving field")
        if batch_solve > 1 and blocks > 1:
            raise Exception("Can't have blocks > 1 if performing simulations in"
                            " batches.")

        self.track = track
        self.wavefront = wavefront
        self.optics_container = optics_container
        self.batch_solve = batch_solve
        self.solver = FieldSolver(wavefront, track)
        self.solver.set_dt(**dt_args, set_bunch=True)

        if compile_solver:
            self.solver = torch.compile(self.solver, mode="reduce-overhead")

        # Define the solver function, depends on if we are batching or using
        # optics propagating.
        if batch_solve > 1:  # Multi solver case
            if self.optics_container is not None:  # Solver uses optics
                def solver_func(index):
                    wvfrt = self.solver.solve_field(1, solve_ends, True, index)
                    self.optics_container.propagate(wvfrt)
                    return wvfrt.get_intensity()
            else:  # No optics solver
                def solver_func(index):
                    wvfrt = self.solver.solve_field(1, solve_ends, True, index)
                    return wvfrt.get_intensity()
            self.solver_func = torch.func.vmap(solver_func)

        else:  # No batch solver case
            if self.optics_container is not None:  # Solver uses optics
                def solver_func(index):
                    wvfrt = self.solver.solve_field(blocks, solve_ends, True,
                                                    index)
                    self.optics_container.propagate(wvfrt)
                    return wvfrt.get_intensity()
            else:  # No optics solver
                def solver_func(index):
                    wvfrt = self.solver.solve_field(blocks, solve_ends, True,
                                                    index)
                    return wvfrt.get_intensity()
            self.solver_func = solver_func

    def solve_incoherent(self, n_part: Optional[int] = None, n_device: int = 1
                         ) -> torch.Tensor:
        """
        Calculates the intensity of incoherent radiation from multiple electrons
        :param n_part: Number of particles to simulate. Defaults to the
         number of tracks simulated. Also, can't be larger than number of
         available tracks. If more than one device is used then the number of
         tracks simulated will be the largest whole number < n_part that is
         divisible by n_device.
        :param n_device: Number of devices to use. If > 1 then
        :return: The intensity as a real 2d torch array
        """

        if n_part > self.track.bunch_r.shape[0]:
            raise Exception("Number of particles must be less than or equal "
                            "to the number of tracks simulated.")
        if n_part < self.batch_solve or n_part % self.batch_solve != 0:
            raise Exception("Number of particles simulated must be N x the "
                            "number of batches, where N is a positive integer")
        if not n_part:
            n_part = self.track.bunch_r.shape[0]

        # Calculate first particle to get the right intensity shape
        idx_0 = 0 if self.batch_solve == 1 else torch.tensor([0])
        intensity_total = torch.zeros_like(self.solver_func(idx_0))

        if n_device == 1:  # Single device case, things are easy
            if self.batch_solve > 1:  # Using batches
                # Make sure number of particles is divisible by batches
                n_part = self.batch_solve * int(n_part / self.batch_solve)
                for batch in range(0, int(n_part / self.batch_solve)):
                    batch_idx = torch.arange(batch * self.batch_solve,
                                             (batch + 1) * self.batch_solve)
                    intensity_total += torch.mean(self.solver_func(batch_idx),
                                                  dim=0)
                intensity_total = intensity_total[0] / int(n_part
                                                           / self.batch_solve)
            else:  # Not using batches
                for index in range(0, n_part):
                    intensity_total += self.solver_func(index)
                intensity_total /= n_part
        else:  # Multi device case is a bit harder
            """
            # Make sure n particles is divisible by number of devices
            n_part = n_device * int(n_part / n_device)

            # Define the intensity tensors on all devices

            intensity_dev = [torch.zeros_like(
                intensity_total, device=torch.device(f"cuda:{i}"))
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
            intensity_total = torch.zeros_like(intensity_total,
                                               device=torch.device("cuda:0"))
            for i, intensity in enumerate(intensity_dev):
                intensity_total += intensity.to("cuda:0")
            """
            raise Exception("Multi GPU case hasn't been implemented yet")
        return intensity_total

    def solve_coherent(self, n_part):
        pass
