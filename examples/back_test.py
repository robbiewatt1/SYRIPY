from SYRIPY import Wavefront, FieldSolver, BeamSolver
from SYRIPY.Optics import FraunhoferProp, CircularAperture, OpticsContainer
from SYRIPY.Tracking import Track, Dipole, FieldContainer
import torch
import matplotlib.pyplot as plt
print("here")

# set defult type to float64
#torch.set_default_dtype(torch.float64)

# Define the magnet setup
d0 = Dipole(torch.tensor([-0., 0., 0.]),      # Location (m)
            1e99,                  # Length (m)
            torch.tensor([0, 1., 0]),   # Field strength (T)
            None,                               # Direction (not implimented yet)
            0.05)                               # Edge length (m)
field = FieldContainer([d0])

# Define the particle track
gamma = torch.tensor([1000.], requires_grad=True)  # Lorentz factor
d0 = torch.tensor([1., 0., 0.], requires_grad=True)            # Initial direction
r0 = torch.tensor([0., 0., 0.], requires_grad=True)      # Initial position (m)
time = torch.linspace(0, 1., 1001)             # Time array samples (ns)
# Define tracking class and track (using c++ implementation, faster but can't
# do gradients)
track = Track(field, device=torch.device("cuda:0"))
track.set_central_params(r0, d0, gamma)
track.sim_central(time)



test = track.beta[-1].sum()
print(test)
test.backward()
print(gamma.grad, d0.grad, r0.grad)
