from SYRIPY import Wavefront, SplitSolver
from SYRIPY.Optics import FraunhoferPropQS, CircularAperture, OpticsContainer
from SYRIPY.Tracking import Track, Dipole, FieldContainer
import torch
import matplotlib.pyplot as plt

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

# define the magnets
dipole_b = 0.5         # Dipole field strength (T)
dipole_l = 0.2         # Dipole length (m)
dipole_edge = 0.05     # Dipole edge length (m)


dipole0 = Dipole(torch.tensor([0, 0, -3]), dipole_l,
                 torch.tensor([0, dipole_b, 0]), None, dipole_edge, device)
dipole1 = Dipole(torch.tensor([0, 0, -0.45]), dipole_l,
                 torch.tensor([0, -dipole_b, 0]), None, dipole_edge, device)
dipole2 = Dipole(torch.tensor([0, 0, 0.6]), dipole_l,
                 torch.tensor([0, -dipole_b, 0]), None, dipole_edge, device)
field = FieldContainer([dipole0, dipole1, dipole2])

gamma = torch.tensor([340. / 0.51099890221])  # Lorentz factor
d0 = torch.tensor([0, 0e-3, 1])               # Initial direction
r0 = torch.tensor([-0.225679, 0, -4])         # Initial position
time = torch.linspace(0, 18, 2001)            # Time array samples (ns)

# Define tracking class and track (using c++ implementation, faster but can't
# do gradients)
track = Track(field, device=device)
track.set_central_params(r0, d0, gamma)
track.sim_central_c(time)
track.plot_track([2, 0])

# Define the initial wavefront
wavefnt = Wavefront(2,           # z position of the wavefront (m)
                    3769200,                      # Radiation angular
                    # frequency (2 pi / ns)
                    [-0.02, 0.02, -0.02, 0.02],  # Wavefront size [x_min,
                    # x_max, y_min, y_max] (m)
                    [512, 512],                # Samples in x and y [n_x, n_y]
                    device=device)               # device used

# Define the optics
aper = CircularAperture(0.02, pad=2)  # Aperture with radiuse 0.02m
prop = FraunhoferPropQS(1,
                        new_shape=[964, 1292],
                        new_bounds=[-0.01875, 0.01875, -0.024225, 0.024225]
                        )       # Fraunhofer propagation to 1m
optics = OpticsContainer([aper, prop])

solver = SplitSolver(wavefnt, track)
solver.set_track(251,              # Number of new samples
                 14,               # Time to split the track
                 10,                # Start of track
                 18,               # End of track
                 plot_track=True)  # Plot updated track

# Solve the field
wave1, wave2 = solver.solve_field()

# Propagate the fields
optics.propagate(wave1)
optics.propagate(wave2)

# Add the fields together
wave1.add_field(wave2)

# Plot the intensity
wave1.plot_intensity()

plt.show()


