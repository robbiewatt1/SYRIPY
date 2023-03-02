import torch
from . import Tracking
from . import Optics
from .FieldSolver import FieldSolver
from .BeamSolver import BeamSolver
from .Wavefront import Wavefront

torch.set_default_tensor_type(torch.DoubleTensor)
