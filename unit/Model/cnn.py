import os
import sys
import torch.nn as nn
import torch.nn.functional as F
import torch
import numpy as np
import pyvista as pv
sys.path.append(os.path.join(os.path.dirname(__file__), "../.."))
from src.Model.cnn import CnnModel, count_parameters


########---------- Simple test with 1 forward pass -----------########
c, d, h, w = 11, 20, 20, 20
torch.manual_seed(3000)
data = torch.randn(1, c, d, h, w)

print(data.size())

#invoke the model
model = CnnModel()
output = model(data)

#compare input and output shapes
print("11 input channels VS 4 output channels")
print("Input dimension -->", data.size())
print("Output dimension -->",output.size())


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
channel = output[0,chan_id,:,:,:].detach().numpy()
p = pv.Plotter(point_smoothing = True)
p.add_volume(np.abs(channel), cmap = "viridis", opacity = "linear")
p.show()
