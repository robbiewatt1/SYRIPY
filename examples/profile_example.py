from SYRIPY import Wavefront, FieldSolver
from SYRIPY.Optics import FraunhoferProp, CircularAperture, OpticsContainer
from SYRIPY.Tracking import Track, Dipole, FieldContainer
import numpy as np
import torch
from torch.profiler import profile, ProfilerActivity
import matplotlib.pyplot as plt

device = torch.device("cuda:1" if torch.cuda.is_available() else "cpu")

# Define the field
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
gamma = torch.tensor([339.3 / 0.51099890221])  # Lorentz factor
d0 = torch.tensor([-0e-3, 0e-3, 1])            # Initial direction
r0 = torch.tensor([-0.0913563873, 0, -1])      # Initial position (m)
time = torch.linspace(0, 14, 1001)             # Time array samples (ns)

moments = np.array([10e-6,     # x (m)
                    20e-6,     # x-xp (m rad)**0.5
                    200e-6,    # xp (rad)
                    100e-6,    # y (m)
                    10e-6,     # y-yp (m rad)**0.5
                    100e-6,    # yp (rad)
                    0.,        # z (m)
                    1.0])**2.  # energy (m_e)

track = Track(field, device=device)
track.set_central_params(r0, d0, gamma)
track.set_beam_params(moments)
track.sim_beam_c(500000, time)

# Define the initial wavefront
wavefnt = Wavefront(2.786062584928902,             # z position of the wavefront (m)
                    3.77e5,                        # Radiation angular frequency (2 pi / ns)
                    [-0.02, 0.02, -0.02, 0.02],    # Wavefront size [x_min, x_max, y_min, y_max] (m)
                    [500, 1],                      # Samples in x and y [n_x, n_y]
                    device=device)                 # device used

solver = FieldSolver(wavefnt, track)
# Set samples along track
solver.set_track(101,       # Number of new samples
                 4.5,         # Start time
                 11.5, flat_power=0.18)        # End time

optics = OpticsContainer([CircularAperture(0.02), FraunhoferProp(1)])


# Batch solver function
def solver_func(index):
    wfnt = solver.solve_field_vmap(index)
    optics.propagate(wfnt)
    return wfnt.get_intensity()


# batch and trace the solver function
batch_func = torch.func.vmap(solver_func)
solver_trace = torch.jit.trace(batch_func, torch.arange(5000))

# Warm up the solver
solver_trace(torch.arange(5000))
solver_trace(torch.arange(5000))


def time_func(intensity):
    for i in range(100):
        intensity += torch.mean(solver_trace(torch.arange(i*5000, (i+1)*5000)),
                                dim=0)
    torch.cuda.synchronize()
    return intensity / 100

intensity = torch.zeros((500, 1), device=device)
with profile(activities=[ProfilerActivity.CPU, ProfilerActivity.CUDA]) as prof:
    time_func(intensity)
print(prof.key_averages().table(sort_by="cuda_time_total"))
