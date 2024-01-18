from SYRIPY import Wavefront, SplitSolver
from SYRIPY.Optics import FraunhoferPropQS, CircularAperture, OpticsContainer
from SYRIPY.Tracking import Track, Dipole, FieldContainer

import torch
import matplotlib.pyplot as plt

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# This example builds on the basic_example.py example. For simulation with a
# relatively large field frequency (e.g. optical and higher), propagation is
# prone to numerical errors unless a large number of wavefront samples are used.
# To avoid this we subtract a quadratic term from the phase of the wavefront
# based on the source location. However, if there are multiple source locations
# the track has to be split into multiple parts. This example demonstrates how
# this can be done.

# define the magnets
dipole_b = 0.5         # Dipole field strength (T)
dipole_l = 0.2         # Dipole length (m)
dipole_edge = 0.05     # Dipole edge length (m)

dipole0 = Dipole(torch.tensor([0, 0, -3]), dipole_l,
                 torch.tensor([0, dipole_b, 0]), None, dipole_edge)
dipole1 = Dipole(torch.tensor([0, 0, -0.45]), dipole_l,
                 torch.tensor([0, -dipole_b, 0]), None, dipole_edge)
dipole2 = Dipole(torch.tensor([0, 0, 0.6]), dipole_l,
                 torch.tensor([0, -dipole_b, 0]), None, dipole_edge)
field = FieldContainer([dipole0, dipole1, dipole2])

gamma = torch.tensor([340. / 0.51099890221])       # Lorentz factor
p0 = torch.tensor([0., 0., 1.]) * gamma * 0.29979  # Initial momentum
r0 = torch.tensor([-0.225679, 0, -4])              # Initial position
time = torch.linspace(0, 18, 1001)                 # Time array samples (ns)

# Define tracking class and track (using c++ implementation, faster but can't
# do gradients)
track = Track(field, device=device)
track.set_central_params(r0, p0)
track.sim_central_c(time)
track.plot_track([2, 0])

# Define the initial wavefront
wavefnt = Wavefront(2,  # z position of the wavefront (m)
                    3.77e6,  # Radiation frequency (2 pi / ns) (green)
                    [-0.02, 0.02, -0.02, 0.02],  # Wavefront size [x_min,
                    # x_max, y_min, y_max] (m)
                    [512, 512],  # Samples in x and y [n_x, n_y]
                    device=device)  # device used

# Define the optics
aper = CircularAperture(0.02, pad=2)  # Aperture with radius 0.02m
prop = FraunhoferPropQS(1,  # Propagation distance (m)
                        new_shape=[964, 1292],  # Wavefront shape after propagation
                        new_bounds=[-0.01875, 0.01875, -0.024225, 0.024225]  # Wavefront bounds after propagation
                        )
optics = OpticsContainer([aper, prop])

solver = SplitSolver(wavefnt, track)
solver.set_track(201,  # Number of new samples
                 14,  # Time to split the track
                 10,  # Start of track
                 18,  # End of track
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
