import os
import sys
import torch.nn as nn
import torch.nn.functional as F
import torch
import numpy as np
import pyvista as pv

sys.path.append(os.path.join(os.path.dirname(__file__), "../.."))
from src.SE3Model.SE3Model import E3nn

########---------- Simple test with 1 forward pass -----------########
c, d, h, w = 11, 20, 20, 20
torch.manual_seed(1000)
data = torch.randn(1, c, d, h, w).cuda()
model = E3nn().cuda()

print("11 input channels VS 6 output channels")
print("Input dimension -->", data.size())


data = torch.einsum('tixyz->txyzi', data)
output = model(data)
output = torch.einsum('txyzi->tixyz', output)
print("Output dimension -->",output.size())


#plot output density map
chan_id = 0 # can be 0,1,2,3
channel = output[0,chan_id,:,:,:].cpu().detach().numpy()
p = pv.Plotter(point_smoothing = True)
p.add_volume(np.abs(channel), cmap = "viridis", opacity = "linear")
p.show()
