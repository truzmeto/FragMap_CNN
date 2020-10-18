import os
import sys
import torch.nn as nn
import torch.nn.functional as F
import torch
import numpy as np
import pyvista as pv

sys.path.append(os.path.join(os.path.dirname(__file__), "../.."))
from src.SE3Model.SE3Unet import UNet
from unit.Util.Shapes3D import get3D_rod
########---------- Simple test with 1 forward pass -----------########

b, c, d, h, w = 2, 11, 96, 96, 96
dim = (b, c, d, h, w)
inp =  get3D_rod(dim).cuda()
torch.manual_seed(1000)


lmax = 0
Rs = list(range(lmax + 1)) 
model = UNet(Rs, inp_channels = c, out_channels = 6).cuda()

print("11 input channels VS 6 output channels")
print("Input dimension -->", inp.size())

data = torch.einsum('tixyz->txyzi', inp)
output = model(data)
output = torch.einsum('txyzi->tixyz', output)
print("Output dimension -->",output.size())


#plot output density map
chan_id = 2 # can be 0,1,2,3
channel = inp[0,chan_id,:,:,:].cpu().detach().numpy()
fs = 16; cmap = 'gist_ncar'#'rainbow'
p = pv.Plotter(point_smoothing = True, shape=(1, 2))
p.subplot(0, 0)
p.add_text("Input", position = 'upper_left', font_size = fs)
p.add_volume(np.abs(channel), cmap = cmap, opacity = "linear")

channel = output[0,chan_id,:,:,:].cpu().detach().numpy()
p.subplot(0, 1)
p.add_text("Output", position = 'upper_left', font_size = fs)
p.add_volume(np.abs(channel), cmap = cmap, opacity = "linear")

p.show()
