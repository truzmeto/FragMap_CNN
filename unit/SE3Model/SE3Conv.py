import os
import sys
import torch.nn as nn
import torch.nn.functional as F
import torch
import numpy as np
import pyvista as pv

sys.path.append(os.path.join(os.path.dirname(__file__), "../.."))
from src.SE3Model.convolution import Convolution

########---------- Simple test with 1 forward pass -----------########
c, d, h, w = 2, 32, 32, 32
torch.manual_seed(1000)
data = torch.randn(1, c, d, h, w).cuda()

Rs_in = [(c, 0)]  
Rs_out = [(c, 0)]  # number of inp and out channels should be same for TransConv

print("SE3 1 forward pass convolution test")
print("Input dimension -->", data.size())


model = Convolution(Rs_in, Rs_out, size=5).cuda()
data = torch.einsum('tixyz->txyzi', data)
output = model(data, trans=False)
output = torch.einsum('txyzi->tixyz', output)

#compare input and output shapes
print("Output dimension -->",output.size())

#plot output density map
chan_id = 0 # can be 0,1,2,3
channel = output[0,chan_id,:,:,:].cpu().detach().numpy()
p = pv.Plotter(point_smoothing = True)
p.add_volume(np.abs(channel), cmap = "viridis", opacity = "linear")
p.show()
