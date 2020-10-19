import os
import sys
import torch.nn as nn
import torch.nn.functional as F
import torch
import numpy as np
import pyvista as pv

sys.path.append(os.path.join(os.path.dirname(__file__), "../.."))
from src.SE3Model.convolution import Convolution
from unit.Util.Shapes3D import get3D_rod
########---------- Simple test with 1 forward pass -----------########

b, c, d, h, w = 2, 2, 32, 32, 32
dim = (b, c, d, h, w)
inp =  get3D_rod(dim).cuda()
torch.manual_seed(1000)


print("SE3 single forward pass convolution test")
print("Input dimension -->", inp.size())

Rs_in = [(c, 0)]  
Rs_out = [(c, 0)]  # number of inp and out channels should be same for TransConv
k_size = 5; pad = k_size//2

model = Convolution(Rs_in, Rs_out, size = k_size, padding = pad).cuda()
data = torch.einsum('tixyz->txyzi', inp)
output = model(data)
output = torch.einsum('txyzi->tixyz', output)
print("Output dimension -->",output.size())

#plot output density maps
chan_id = 0 # can be 0,1,2,3
chan1 = inp[0,chan_id,:,:,:].cpu().detach().numpy()
fs = 16; cmap = 'gist_ncar'#'rainbow'
p = pv.Plotter(point_smoothing = True, shape=(1, 2))
p.subplot(0, 0)
p.add_text("Input", position = 'upper_left', font_size = fs)
p.add_volume(np.abs(chan1), cmap = cmap, opacity = "linear")

chan2 = output[0,chan_id,:,:,:].cpu().detach().numpy()
p.subplot(0, 1)
p.add_text("Output", position = 'upper_left', font_size = fs)
p.add_volume(np.abs(chan2), cmap = cmap, opacity = "linear")

p.show()
