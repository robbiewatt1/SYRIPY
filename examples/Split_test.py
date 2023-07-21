from ERS import Wavefront, FieldSolver, BeamSolver, SplitSolver
from ERS.Optics import FraunhoferProp, CircularAperture, OpticsContainer, ThinLens, FresnelProp, TestProp
from ERS.Tracking import Track, Dipole, FieldContainer
import torch
import matplotlib.pyplot as plt

torch.set_default_tensor_type(torch.DoubleTensor)


device = torch.device("cuda:1" if torch.cuda.is_available() else "cpu")


dipole0 = Dipole(torch.tensor([0, 0, 0]), 0.203274830142196,
                 torch.tensor([0, 0.49051235, 0]), None, 0.05, device)
dipole1 = Dipole(torch.tensor([0, 0, 1.3334]), 0.203274830142196,
                 torch.tensor([0, -0.49051235, 0]), None, 0.05, device)
dipole2 = Dipole(torch.tensor([0, 0, 2.6668]), 0.203274830142196,
                 torch.tensor([0, -0.49051235, 0]), None, 0.07, device)
field = FieldContainer([dipole1, dipole2])

gamma = 339.3 / 0.51099890221              # Lorentz factor
d0 = torch.tensor([1.115e-1, 0e-3, 1])        # Initial direction
r0 = torch.tensor([-0.259, 0, -1])  # Initial position
time = torch.linspace(0, 18, 10001)         # Time array samples (ns)

# Define tracking class and track (using c++ implementation, faster but can't
# do gradients)
track = Track(field, device=device)
track.sim_single_c(time, r0, d0, gamma)
track.plot_track([2, 0])

# Define the initial wavefront
wavefnt = Wavefront(5,           # z position of the wavefront (m)
                    3769200,                      # Radiation angular frequency (2 pi / ns)
                    [-0.04, 0.04, -0.04, 0.04],  # Wavefront size [x_min, x_max, y_min, y_max] (m)
                    [512, 512],                # Samples in x and y [n_x, n_y]
                    device=device)               # device used

solver = SplitSolver(wavefnt, track)
solver.set_track(10, 251,       # Number of new samples
              5,         # Start time
              14, flat_power=0.2, mode="nn")

wave1, wave2 = solver.solve_field(5, plot_track=True)

fig, ax = plt.subplots()
ax.pcolormesh(wave1.get_intensity().cpu().T)

prop = TestProp(100)
prop.propagate(wave1)
prop.propagate(wave2)
wave1.add_field(wave2)
wave1.plot_intensity()

plt.show()