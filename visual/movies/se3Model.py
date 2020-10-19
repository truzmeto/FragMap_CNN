"""
make a movie to show equivariance!
"""

import os
import sys
import torch.nn as nn
import torch.nn.functional as F
import torch
import numpy as np
import pyvista as pv
from scipy import ndimage

sys.path.append(os.path.join(os.path.dirname(__file__), "../.."))
from unit.Util.Shapes3D import get3D_rod
from src.SE3Model.SE3Model import E3nn

def rotate_ligand(ligand, rotation_angle):    
    
    ligand = ndimage.interpolation.rotate(ligand,
                                          angle = rotation_angle,
                                          axes=(2,3),
                                          reshape=False,
                                          order=1,
                                          mode= 'nearest',#'constant',
                                          cval=0.0)
    
    return ligand



########---------- Simple test with 1 forward pass -----------########

b, c, d, h, w = 1, 1, 32, 32, 32
dim = (b, c, d, h, w)
torch.manual_seed(1000)


Rs_in = [(c, 0)]  
Rs_out = [(c, 0)]  # number of inp and out channels should be same for TransConv

#plot output density maps
chan_id = 0 # can be 0,1,2,3
fs = 16; cmap = 'gist_ncar'#'rainbow'

model = E3nn(num_input_channels=1, lmax=0).cuda()
inp =  get3D_rod(dim).cuda()

# Open a movie file
p = pv.Plotter(point_smoothing = True, shape=(1, 2))
p.open_movie('my_movie.mp4')


#calc conv for original
data = torch.einsum('tixyz->txyzi', inp)
output = model(data)
output = torch.einsum('txyzi->tixyz', output)

for i in range(180):

    inpR = rotate_ligand(inp.cpu().detach().numpy(), rotation_angle= i*2.0)
    inpR = torch.from_numpy(inpR).float().to('cuda')
    
    dataR = torch.einsum('tixyz->txyzi', inpR)
    outputR = model(dataR)
    outputR = torch.einsum('txyzi->tixyz', outputR)

    outputUR = rotate_ligand(outputR.cpu().detach().numpy(), rotation_angle= -i*2.0)
    outputUR = torch.from_numpy(outputUR).float().to('cuda')
    err = (output - outputUR).pow(2).mean().sqrt()
    err = err.item()
    
    p.subplot(0, 0)
    chan1 = inpR[0,chan_id,:,:,:].cpu().detach().numpy()
    p.add_text("Input ", position = 'lower_left', font_size = fs)
    p.add_text("RMS(out - out_unrot) =  " + str(round(err,5)), position = 'upper_left', font_size = fs-2)
    p.add_volume(np.abs(chan1), cmap = cmap, opacity = "linear", show_scalar_bar=False)
    
    p.subplot(0, 1)
    chan2 = outputR[0,chan_id,:,:,:].cpu().detach().numpy()
    p.add_text("Output", position = 'lower_left', font_size = fs)
    p.add_volume(np.abs(chan2), cmap = cmap, opacity = "linear", show_scalar_bar=False)

    if i == 0 :
        p.show(auto_close=False)

    p.write_frame()
    p.clear()

p.close()
