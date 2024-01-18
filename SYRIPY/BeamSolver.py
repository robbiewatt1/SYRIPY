import torch
from .Wavefront import Wavefront
from .Tracking import Track
from .Optics import OpticsContainer
from typing import Optional


class BeamSolver:
    """
    Class for calculating the intensity of a beam of particles using either a
    convolution or monte carlo method.
    """

    def __init__(self, wavefront: Wavefront, track: Track,
                 optics: OpticsContainer = None) -> None:
        """
        :param wavefront: Instance of Wavefront class
        :param track: Instance of Track class (should already have calculated
         single or bunch tracks)
        :param optics: Instance of OpticsContainer class.
        """

        self.track = track
        self.wavefront = wavefront
        self.optics = optics

    def convolve_beam(self, sigma_x: torch.Tensor, sigma_y: torch.Tensor
                      ) -> torch.Tensor:
        """
        Calculates the intensity of incoherent radiation from multiple electrons
        using a convolution method.
        :param init_beam_params: Initial beam parameters [x, x-xp, xp, y, y-yp,
            yp, z, gamma]**2
        :return: The intensity as a real 2d torch array
        """
        '''
        if init_beam_params is not None:
            self.track.set_beam_params(init_beam_params)

        # First calculate blurring kernel
        electron_transport = self.track.get_transport_matrix(
            self.track.r[2, 0], self.wavefront.source_location[2])
        photon_transport = self.optics.get_propagation_matrix().to(
            self.track.device)

        transport = photon_transport @ electron_transport
        beam_matrix = self.track.get_beam_matrix()
        beam_matrix = transport @ beam_matrix @ transport.T
        sigma_x, sigma_y = beam_matrix[0, 0]**0.5, beam_matrix[3, 3]**0.5
        print(sigma_x, sigma_y)
        '''
        sigma_x, sigma_y = sigma_x.to(self.wavefront.device), sigma_y.to(
            self.wavefront.device)

        # Now perform convolution
        intensity = self.wavefront.get_intensity()

        # Pad the wavefront to avoid edge effects
        kernel_size_x = int(5 * sigma_x / self.wavefront.delta[0])
        kernel_size_y = int(5 * sigma_y / self.wavefront.delta[1])
        intensity = torch.nn.functional.pad(
            intensity, (kernel_size_y, kernel_size_y,
                        kernel_size_x, kernel_size_x))
        kernel_x_axis = torch.linspace(-5 * sigma_x.item(), 5 * sigma_x.item(),
                              2 * kernel_size_x + 1,
                                       device=self.wavefront.device)
        kernel_y_axis = torch.linspace(-5 * sigma_y.item(), 5 * sigma_y.item(),
                              2 * kernel_size_y + 1,
                                       device=self.wavefront.device)
        kernel = (torch.exp(-0.5 * (kernel_x_axis / sigma_x)**2.)[:, None]
                  * torch.exp(-0.5 * (kernel_y_axis / sigma_y)**2.)[None, :])
        intensity = torch.nn.functional.conv2d(
            intensity[None, None, ...], kernel[None, None, ...]
        )
        return intensity[0, 0]


    def monte_carlo_beam(self, n_part: Optional[int] = None, n_device: int = 1
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

        '''
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
        '''
        raise Exception("Monte Carlo solver hasn't been implemented yet")


