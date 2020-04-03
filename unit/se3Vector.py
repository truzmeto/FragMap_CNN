import torch

from se3cnn.image.convolution import SE3Convolution
#from se3cnn import SE3Convolution
from se3cnn.image.gated_block import GatedBlock
#from se3cnn.blocks import GatedBlock

size = 32  # space size

scalar_field = torch.randn(1, 1, size, size, size)  # [batch, _, x, y, z]

Rs_in = [(2,0), (2, 1)]  # 1 scalar field
Rs_out = [(2,0), (2, 1)]  # 1 vector field

block = GatedBlock((1,0), (2, 2), size=5)
conv = SE3Convolution(Rs_out, Rs_in, size=5)
# conv.weight.size() == [2] (2 radial degrees of freedom)

# vector_field = conv(scalar_field)  # [batch, vector component, x, y, z]
vector_field = block(scalar_field)
scalar_field = conv(vector_field)

print(vector_field.size())
