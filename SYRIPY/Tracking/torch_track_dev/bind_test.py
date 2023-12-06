
import torch
from Magnets import Dipole, FieldContainer
import cTrack
import time as timer
import matplotlib.pyplot as plt



# Define the magnet setup
d0 = Dipole(torch.tensor([-0., 0., 0.]),      # Location (m)
            1e99,                  # Length (m)
            torch.tensor([0, 1., 0]),   # Field strength (T)
            None,                               # Direction (not implimented yet)
            0.05)                               # Edge length (m)
field = FieldContainer([d0])
field = field.gen_c_container()





class DiffTrack(torch.autograd.Function):
    """
    This class makes the differentiable track"""

    def __init__(self):
        super(DiffTrack, self).__init__()
    
    @staticmethod
    def forward(ctx, position, momentum, track):
        """
        This function takes the inputs and the track and returns the output
        """
        r_track = cTrack.ThreeVector(position.tolist(), True)
        m_track = cTrack.ThreeVector(momentum.tolist(), True)
        track.setCentralInit(r_track, m_track)
        result = track.simulateTrack()
        pos, beta = torch.tensor(result[1]), torch.tensor(result[2])
        ctx.track = track
        return pos, beta
    
    @staticmethod
    @torch.autograd.function.once_differentiable
    def backward(ctx, position_grad_out, beta_grad_out):
        """
        This function takes the gradient and returns the gradient of the inputs
        """
        track = ctx.track
        (position_grad, momentum_grad) = track.backwardTrack(position_grad_out, beta_grad_out)
        return torch.tensor(position_grad), torch.tensor(momentum_grad), None
    
def dt(poisition, momentum):
    return DiffTrack.apply(poisition, momentum, track)


track = cTrack.Track()
track.setField(field)
track.setTime(0, 1., 1001)

gamma = torch.tensor([1000.], requires_grad=True)  # Lorentz factor
d0 = torch.tensor([1, 0., 0.], requires_grad=True)            # Initial direction
r0 = torch.tensor([0., 0., 0.], requires_grad=True)      # Initial position (m)
p0 = (0.299792458 * (gamma**2.0 - 1.)**0.5 * d0 / torch.norm(d0))
print("start")
pos, beta = dt(r0, p0)

test = beta[-1].sum()
print(test)
test.backward()
print(gamma.grad, d0.grad, r0.grad)

