import os
import sys
import torch.nn as nn
import torch.nn.functional as F
import torch
import numpy as np
import pyvista as pv
sys.path.append(os.path.join(os.path.dirname(__file__), ".."))
from src.cnn import CnnModel, count_parameters


########---------- Simple test with 1 forward pass -----------########
c, d, h, w = 11, 30, 30, 30
torch.manual_seed(3000)
data = torch.randn(1, c, d, h, w)
    
#invoke the model
model = CnnModel()
output = model(data)

#compare input and output shapes
print("11 input channels VS 4 output channels")
print("Input dimension -->", data.size())
print("Output dimension -->",output.size())


#plot output density map
chan_id = 0 # can be 0,1,2,3
channel = output[0,chan_id,:,:,:].detach().numpy()
#channel = -0.6*np.log(channel+0.01)
p = pv.Plotter(point_smoothing = True)
p.add_volume(channel, cmap = "viridis", opacity = "linear")
p.show()

  
print(count_parameters(model))
   