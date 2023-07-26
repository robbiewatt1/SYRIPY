from ERS import Wavefront, FieldSolver, BeamSolver, SplitSolver
from ERS.Optics import FraunhoferProp, CircularAperture, OpticsContainer, ThinLens, FresnelProp, TestProp
from ERS.Tracking import Track, Dipole, FieldContainer
import torch
import matplotlib.pyplot as plt

torch.set_default_tensor_type(torch.DoubleTensor)


device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")


dipole0 = Dipole(torch.tensor([0, 0, -3]), 0.203274830142196,
                 torch.tensor([0, 0.49051235, 0]), None, 0.05, device)
dipole1 = Dipole(torch.tensor([0, 0, 1.3334-3]), 0.203274830142196,
                 torch.tensor([0, -0.49051235, 0]), None, 0.05, device)
dipole2 = Dipole(torch.tensor([0, 0, 2.6668-3]), 0.203274830142196,
                 torch.tensor([0, -0.49051235, 0]), None, 0.05, device)
field = FieldContainer([dipole0, dipole1, dipole2])

gamma = 339.3 / 0.51099890221              # Lorentz factor
d0 = torch.tensor([0, 0e-3, 1])        # Initial direction
r0 = torch.tensor([-0.14916-9.04e-6, 0, -1-3])  # Initial position
time = torch.linspace(0, 18, 10001)         # Time array samples (ns)

# Define tracking class and track (using c++ implementation, faster but can't
# do gradients)
track = Track(field, device=device)
track.sim_single_c(time, r0, d0, gamma)
track.plot_track([2, 0])

# Define the initial wavefront
wavefnt = Wavefront(5,           # z position of the wavefront (m)
                    376920,                      # Radiation angular
                    # frequency (2 pi / ns)
                    [-0.05, 0.05, -0.05, 0.05],  # Wavefront size [x_min,
                    # x_max, y_min, y_max] (m)
                    [512, 512],                # Samples in x and y [n_x, n_y]
                    device=device)               # device used
prop1 = TestProp(40, pad=2)
prop2 = FresnelProp(40, pad=8)

solver = SplitSolver(wavefnt, track)
solver.set_track(10, 251, 6, 14, flat_power=0.2, mode="nn", plot_track=True)
wave1, wave2 = solver.solve_field()
prop1.propagate(wave1)
prop1.propagate(wave2)
#wave1.add_field(wave2)
wave1.plot_intensity()
wave2.plot_intensity()


wave1, wave2 = solver.solve_field()
prop2.propagate(wave1)
prop2.propagate(wave2)
#wave1.add_field(wave2)
wave1.plot_intensity()

plt.show()

track.sim_single_c(time, r0, d0, gamma)
solver = FieldSolver(wavefnt, track)
solver.set_track(251, 5, 14, flat_power=0.2, mode="nn")
wave1 = solver.solve_field()
wave1.plot_intensity()

prop1.propagate(wave1)
wave1.plot_intensity()
wave1 = solver.solve_field()
prop2.propagate(wave1)
wave1.plot_intensity(ds_fact=4)

plt.show()
