from ERS import Wavefront, FieldSolver
from ERS.Optics import FraunhoferProp, CircularAperture, OpticsContainer
from ERS.Tracking import Track, Dipole, FieldContainer
import numpy as np
import torch
from torch.profiler import profile, ProfilerActivity

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

# Define the field
dipole0 = Dipole(torch.tensor([0, 0, 0]), 0.203274830142196,
                 torch.tensor([0, 0.49051235, 0]), None, 0.05, device)
dipole1 = Dipole(torch.tensor([0, 0, 1.0334]), 0.203274830142196,
                 torch.tensor([0, -0.49051235, 0]), None, 0.05, device)
dipole2 = Dipole(torch.tensor([0, 0, 2.0668]), 0.203274830142196,
                 torch.tensor([0, -0.49051235, 0]), None, 0.05, device)
field = FieldContainer([dipole0, dipole1, dipole2])

# Define the central track and beam moments
gamma = torch.tensor([339.3 / 0.51099890221])              # Lorentz factor
d0 = torch.tensor([-1e-3, 0e-3, 1])        # Initial direction
r0 = torch.tensor([-0.1155863873, 0, -1])  # Initial position
moments = np.array([10e-6, # x (m)
                    20e-6,  # x-xp (m rad)**0.5
                    200e-6, # xp (rad)
                    100e-6, # y (m)
                    10e-6,  # y-yp (m rad)**0.5
                    100e-6, # yp (rad)
                    1.0     # energy (m_e)
                   ])**2.
time = torch.linspace(0, 14, 1001)         # Time array samples (ns)

track = Track(field, device=device)
track = torch.jit.script(track)
track.sim_bunch_c(10000, time, r0, d0, gamma, moments)

# Define the initial wavefront
wavefnt = Wavefront(2.786062584928902,             # z position of the wavefront (m)
                    3.77e5,                        # Radiation angular frequency (2 pi / ns)
                    [-0.02, 0.02, -0.02, 0.02],    # Wavefront size [x_min, x_max, y_min, y_max] (m)
                    [500, 1],                    # Samples in x and y [n_x, n_y]
                    device=device)                 # device used

solver = FieldSolver(wavefnt, track)
# Set samples along track
solver.set_dt(100,       # Number of new samples
              4.5,         # Start time
              11.5, flat_power=0.18)        # End time

optics = OpticsContainer([CircularAperture(0.02), FraunhoferProp(1)])

def solver_func(index):
    wfnt = solver.solve_field_vmap(index)
    optics.propagate(wfnt)
    return wfnt.get_intensity()

batch_func = torch.func.vmap(solver_func)
solver_trace = torch.jit.trace(batch_func, torch.arange(5000))
intens = solver_trace(torch.arange(5000))
intens = solver_trace(torch.arange(5000))


def time_func():
    for i in range(1000):
        intens = solver_trace(torch.arange(5000))
    torch.cuda.synchronize()
    return intens

with profile(activities=[ProfilerActivity.CPU, ProfilerActivity.CUDA]) as prof:
    time_func()
print(prof.key_averages().table(sort_by="cuda_time_total"))
