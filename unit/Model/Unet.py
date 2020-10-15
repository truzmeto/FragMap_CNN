import os
import sys
import torch
import numpy as np
import pyvista as pv

sys.path.append(os.path.join(os.path.dirname(__file__), "../.."))
from src.Model.Unet import UNet

########---------- Simple test with 1 forward pass -----------########

c, d, h, w = 11, 96, 96, 96
torch.manual_seed(3000)
data = torch.randn(1, c, d, h, w)
print(data.size())

#invoke the model
in_dim = c
out_dim = 6

model = UNet(in_dim=in_dim, out_dim=out_dim, num_filters=4)
output = model(data)

#compare input and output shapes
print("11 input channels VS 6 output channels")
print("Input dimension -->", data.size())
print("Output dimension -->", output.size())


#plot output density map
chan_id = 0 # can be 0,1,2,3
channel = output[0,chan_id,:,:,:].detach().numpy()
p = pv.Plotter(point_smoothing = True)
p.add_volume(np.abs(channel), cmap = "viridis", opacity = "linear")
p.show()
