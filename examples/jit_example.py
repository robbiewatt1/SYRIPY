from SYRIPY import Wavefront, FieldSolver
from SYRIPY.Tracking import Track, Dipole, FieldContainer

import torch
import time as timer

# This example shows how SYRIPY simulations can be accelerated using
# torch.jit.script. Usually the most expensive part of the simulation is the
# calculation of the initial field. We can use just in time compilation to
# speed up this part of the simulation. This example shows how this can be done.
# I find a speed-up of about 4x on my machine.

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Define the magnet setup
d0 = Dipole(torch.tensor([-0.116, 0, 0]),  # Location (m)
            0.203,  # Length (m)
            torch.tensor([0, 0.491, 0]),  # Field strength (T)
            None,  # Direction (not implemented yet)
            0.05)  # Edge length (m)
d1 = Dipole(torch.tensor([0, 0, 1.033]), 0.203,
            torch.tensor([0, -0.491, 0]), None, 0.05)
d2 = Dipole(torch.tensor([0, 0, 2.067]), 0.203,
            torch.tensor([0, -0.491, 0]), None, 0.05)
field = FieldContainer([d0, d1, d2])

# Define the particle track
gamma = torch.tensor([340. / 0.51099890221])       # Lorentz factor
p0 = torch.tensor([0., 0., 1.]) * gamma * 0.29979   # Initial momentum
r0 = torch.tensor([-0.091356, 0, -1])           # Initial position (m)
time = torch.linspace(0, 14, 1001)  # Time array samples (ns)

# Define tracking class and track (using c++ implementation, faster but can't
# do gradients)
track = Track(field, device=device)
track.set_central_params(r0, p0)
track.sim_central_c(time)

# Define the initial wavefront
wavefnt = Wavefront(3.0,  # z position of the wavefront (m)
                    3.77e5,  # Radiation angular frequency (2 pi / ns)
                    [-0.02, 0.02, -0.02, 0.02],  # Wavefront size [x_min, x_max, y_min, y_max] (m)
                    [300, 300],  # Samples in x and y [n_x, n_y]
                    device=device)  # device used

# Define the field solver class
solver = FieldSolver(wavefnt, track)

# Select the track region of interest and redistribution the samples along the
# track
solver.set_track(101,  # Number of new samples
                 4,  # Start time
                 13,)  # End time

# time how long it takes to run 1000 simulations without JIT
start_time = timer.time()
for i in range(1000):
    _ = solver.solve_field()
torch.cuda.synchronize()
end_time = timer.time()
print("No JIT: ", (end_time - start_time) / 1000,
      "s per simulation")

# Now we can use torch.jit.script to compile the solve_field function
solver = torch.jit.script(solver)

# Since we are using jit we need to run the simulation a couple of times to
# compile the function
_ = solver.solve_field()
_ = solver.solve_field()

# Now time how long it takes to run 1000 simulations with JIT
start_time = timer.time()
for i in range(1000):
    _ = solver.solve_field()
torch.cuda.synchronize()
end_time = timer.time()
print("With JIT: ", (end_time - start_time) / 1000,
      "s per simulation")
