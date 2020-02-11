from functools import partial
import torch
from e3nn.radial import CosineBasisModel
from e3nn.kernel import Kernel
from e3nn.point.operations import Convolution
from e3nn.util.plot import plot_sh_signal
import matplotlib.pyplot as plt

# Radial model:  R -> R^d
# Projection on cos^2 basis functions followed by a fully connected network
RadialModel = partial(CosineBasisModel, max_radius=3.0, number_of_basis=3, h=100, L=1, act=torch.relu)

# kernel: composed on a radial part that contains the learned parameters
#  and an angular part given by the spherical hamonics and the Clebsch-Gordan coefficients
K = partial(Kernel, RadialModel=RadialModel)

# Define input and output representations
Rs_in = [(1, 0)]  # one scalar
Rs_out = [(1, l) for l in range(10)]

# Use the kernel to define a convolution operation
conv = Convolution(K, Rs_in, Rs_out)

n = 3  # number of points
features = torch.ones(1, n, 1)
geometry = torch.randn(1, n, 3)

features = conv(features, geometry)
print(features)
