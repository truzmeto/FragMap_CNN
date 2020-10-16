import sys
import os
import torch
import pyvista as pv
import numpy as np

sys.path.append(os.path.join(os.path.dirname(__file__), "../.."))

def get3D_rod(dim = (1,1,56,56,56) ):
    """
    function that constructs xyz axiz type object
    with diff. colored tips to check if rotations
    are physcal
    """

    volume = torch.zeros(dim)
    length = dim[2] // 2  - 6
    ct = [dim[2]//2, dim[3]//2, dim[4]//2]

    volume[:,:,ct[0]:ct[0]+length,ct[1],ct[2]] = 0.5
    volume[:,:,ct[0]+length:ct[0]+length+2,ct[1],ct[2]] = 0.2

    volume[:,:,ct[0],ct[1]:ct[1]+length,ct[2]] = 0.5
    volume[:,:,ct[0],ct[1]+length:ct[1]+length+2,ct[2]] = 1.
    
    volume[:,:,ct[0],ct[1],ct[2]:ct[2]+length] = 0.5
    volume[:,:,ct[0],ct[1],ct[2]+length:ct[2]+length+2] = 2.0
    
    volume[:,:,ct[0],ct[1]:ct[1]+length,ct[2]:ct[2]+length] = 0.2
    volume[:,:,ct[0],ct[1]+length:ct[1]+length+1,ct[2]+length:ct[2]+length+1] = 1.5

    return volume

if __name__=='__main__':
   
    volume =  get3D_rod()

    p = pv.Plotter(point_smoothing = True)
    fs = 15
    channel = volume[0,0,:,:,:].cpu().numpy()
    text = '3D coordinate system & yz-plane'
    p.add_text(text, position = 'upper_left', font_size = fs)
    p.add_volume(channel, cmap = "viridis_r", opacity = "linear")
    p.show()
        
