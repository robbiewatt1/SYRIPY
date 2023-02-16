from ERS import Wavefront, FieldSolver
from ERS.Optics import FraunhoferProp, CircularAperture
from ERS.Tracking import Track, Dipole, FieldContainer
import torch
import matplotlib.pyplot as plt

# Check for gpu
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

# Define field
d1 = Dipole([0, 0, 0], 0.203274830142196, -0.49051235, None, 0.01)
d2 = Dipole([0, 0, 1.0334], 0.203274830142196, -0.49051235, None,
            0.01)
field = FieldContainer([d1, d2])

# define electron properties
gamma = 339.3 / 0.51099890221
d0 = torch.tensor([0.09313368161783511, 0, 1])
r0 = torch.tensor([-0.09311173301, 0, -1])

# Solve the track
track = Track(device=device)
time = torch.linspace(0, 10, 10000)
track.sim_single_c(field, time, r0, d0, gamma)

# Plot the track
track.plot_track(axes=[2, 0])

# Define the wavefront and field solver
wavefnt = Wavefront(1.7526625849289021, 3.77e5,
                    [-0.01, 0.01, -0.01, 0.01],
                    [1000, 1000], device=device)
solver = FieldSolver(wavefnt, track, device=device)

# Reduce the time samples to 200
solver.set_dt(200)

# Solve the field
solver.solve()

# Plot the intensity
wavefnt.plot_intensity()

# Propagate through an aperture and to focus
aper = CircularAperture(0.01)
aper.propagate(wavefnt)
prop = FraunhoferProp(0.105)
prop.propagate(wavefnt, new_shape=[2000, 2000],
               new_bounds=[-0.0015, 0.0015, -0.0015, 0.0015])

# Plot the intensity again
wavefnt.plot_intensity()
plt.show()
