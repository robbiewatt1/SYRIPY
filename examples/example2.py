from ERS import Wavefront, FieldSolver, BeamSolver
from ERS.Optics import FraunhoferProp, CircularAperture, OpticsContainer
from ERS.Tracking import Track, Dipole, FieldContainer
import torch
import matplotlib.pyplot as plt
import numpy as np
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

# Define the magnet setup
d0 = Dipole(torch.tensor([-0.1156, 0, 0]),      # Location (m)
            0.203274830142196,                  # Length (m)
            torch.tensor([0, 0.49051235, 0]),   # Field strength (T)
            None,                               # Direction (not implimented yet)
            0.05)                               # Edge length (m)
d1 = Dipole(torch.tensor([0, 0, 1.0334]), 0.203274830142196,
            torch.tensor([0, -0.49051235, 0]), None, 0.05)
d2 = Dipole(torch.tensor([0, 0, 2.0668]), 0.203274830142196,
            torch.tensor([0, -0.49051235, 0]), None, 0.05)
field = FieldContainer([d0, d1, d2])

# Define the central track and beam moments
gamma = 339.3 / 0.51099890221              # Lorentz factor
d0 = torch.tensor([-0e-3, 0e-3, 1])        # Initial direction
r0 = torch.tensor([-0.1155863873, 0, -1])  # Initial position
moments = np.array([100e-6, # x (m)
                    10e-6,  # x-xp (m rad)**0.5
                    300e-6, # xp (rad)
                    100e-6, # y (m)
                    10e-6,  # y-yp (m rad)**0.5
                    100e-6, # yp (rad)
                    5.0     # energy (m_e)
                   ])**2.
time = torch.linspace(0, 14, 1000)         # Time array samples (ns)

track = Track(device=device)
track.sim_bunch_c(10000, field, time, r0, d0, gamma, moments)

# Plot track
fig, ax = track.plot_bunch([2, 0], 10)
ax.set_xlabel("z (m)")
ax.set_ylabel("x (m)")

# Define the initial wavefront
wavefnt = Wavefront(2.786062584928902,             # z position of the wavefront (m)
                    3.77e5,                        # Radiation angular frequency (2 pi / ns)
                    [-0.02, 0.02, -0.02, 0.02],    # Wavefront size [x_min, x_max, y_min, y_max] (m)
                    [500, 1],                      # Samples in x and y [n_x, n_y]
                    device=device)                 # device used

# Define the optics
aper = CircularAperture(0.02)  # Aperture with radiuse 0.02m
prop = FraunhoferProp(1)       # Fraunhofer propagation to 1m
optics = OpticsContainer([aper, prop])

# Define the field solver class
solver = BeamSolver(wavefnt, track, optics,
                    dt_args={"new_samples": 200, # Number of new samples
                             "t_start": 4,       # Start time
                             "t_end": 13},       # End time
                    batch_solve=2500             # Number of simulations per batch
                   )
fig, ax = track.plot_bunch([2, 0], 10)
ax.set_xlabel("z (m)")
ax.set_ylabel("x (m)")

intesity = solver.solve_incoherent(10000)  # Number of particles to simulate
fig, ax = plt.subplots()
ax.plot(intesity.cpu())
plt.show()
