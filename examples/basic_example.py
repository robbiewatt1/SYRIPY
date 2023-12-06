from SYRIPY import Wavefront, FieldSolver, BeamSolver
from SYRIPY.Optics import FraunhoferPropQS, CircularAperture, OpticsContainer
from SYRIPY.Tracking import Track, Dipole, FieldContainer
import torch
import matplotlib.pyplot as plt

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

# Define the particle track
gamma = torch.tensor([339.3 / 0.51099890221])      # Lorentz factor
p0 = torch.tensor([0., 0., 1.]) * gamma * 0.29979  # Initial momentum
p0.requires_grad = True
r0 = torch.tensor([-0.0913563873, 0, -1], requires_grad=True)          # Initial position (m)
time = torch.linspace(0, 14, 1001)                 # Time array samples (ns)

# Define tracking class and track (using c++ implementation, faster but can't
# do gradients)
track = Track(field, device=device)
track.set_central_params(r0, p0)
track.sim_central_c(time)

# Plot track
fig, ax = track.plot_track([2, 0])
ax.set_xlabel("z (m)")
ax.set_ylabel("x (m)")

# Define the initial wavefront
wavefnt = Wavefront(2.786062584928902,             # z position of the wavefront (m)
                    3.77e5,                        # Radiation angular frequency (2 pi / ns)
                    [-0.02, 0.02, -0.02, 0.02],    # Wavefront size [x_min, x_max, y_min, y_max] (m)
                    [250, 250],                    # Samples in x and y [n_x, n_y]
                    device=device)                 # device used

# Define the optics
aper = CircularAperture(0.02, pad=2)  # Aperture with radiuse 0.02m
prop = FraunhoferPropQS(1)       # Fraunhofer propagation to 1m
optics = OpticsContainer([aper, prop])

# Define the field solver class
solver = FieldSolver(wavefnt, track)
# Set samples along track
solver.set_track(201,       # Number of new samples
                 4,         # Start time
                 13)        # End time

fig, ax = track.plot_track(axes=[2, 0])
ax.set_xlabel("z (m)")
ax.set_ylabel("x (m)")

print("start")
# Solve the field
solver.solve_field()

# Plot the intensity
wavefnt.plot_intensity()

# Propagate the wavefront
optics.propagate(wavefnt)
wavefnt.plot_intensity()

intensity = wavefnt.get_intensity().sum().backward()
print(r0.grad, p0.grad)
print("end")
plt.show()
