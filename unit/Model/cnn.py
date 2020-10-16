import os
import sys
import torch.nn as nn
import torch.nn.functional as F
import torch
import numpy as np
import pyvista as pv
sys.path.append(os.path.join(os.path.dirname(__file__), "../.."))
from src.Model.cnn import CnnModel, count_parameters
from unit.Util.Shapes3D import get3D_rod

########---------- Simple test with 1 forward pass -----------########

b, c, d, h, w = 1, 11, 32, 32, 32
dim = (b, c, d, h, w)
data =  get3D_rod(dim).cuda()
torch.manual_seed(1000)

#invoke the model
model = CnnModel().cuda()
output = model(data)

print("Input dimension -->", data.size())
print("Output dimension -->", output.size())


# estimate number of parameters
est_params = count_parameters(model)
n_layer = [11, 48, 24, 12, 6]
n_params = 0
k_size = 3
for i in range(len(n_layer)-1):
    if i < 1:
        n_params += (k_size**3*n_layer[i]+ 1 )*n_layer[i+1]
    else:
        n_params += (k_size**3*n_layer[i]+ 3 )*n_layer[i+1]

print("Number of parameters in model", est_params)
print("Number of parameteres estimated", n_params)   
if(est_params != n_params):
    print("Check ConvNet structure: kernel size, and # of Conv units in each layer")


#plot output density map
chan_id = 0 # can be 0,1,2,3
channel = data[0,chan_id,:,:,:].cpu().detach().numpy()
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

