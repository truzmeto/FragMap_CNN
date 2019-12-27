import torch
from TorchProteinLibrary.Volume import VolumeRotation
from TorchProteinLibrary.FullAtomModel import getRandomRotation, getRandomTranslation
import pyvista as pv
import numpy as np
import sys
import os

sys.path.append(os.path.join(os.path.dirname(__file__), ".."))
from src.rot24 import Rot90Seq

def get3D_rod():
    """
    function that constructs xyz axiz type object
    with diff. colored tips to check if rotations
    are physcal
    """

    volume = torch.zeros(1,1,55,54,53)
    length = 15
    st = [27,26,25]

    volume[:,:,st[0]:st[0]+length,st[1],st[2]] = 0.5
    volume[:,:,st[0]+length:st[0]+length+2,st[1],st[2]] = 0.2

    volume[:,:,st[0],st[1]:st[1]+length,st[2]] = 0.5
    volume[:,:,st[0],st[1]+length:st[1]+length+2,st[2]] = 1.
    
    volume[:,:,st[0],st[1],st[2]:st[2]+length] = 0.5
    volume[:,:,st[0],st[1],st[2]+length:st[2]+length+2] = 2.0
    
    volume[:,:,st[0],st[1]:st[1]+length,st[2]:st[2]+length] = 0.2
    volume[:,:,st[0],st[1]+length:st[1]+length+1,st[2]+length:st[2]+length+1] = 1.5

    return volume


if __name__=='__main__':
   
    volume =  get3D_rod()
    R = getRandomRotation(1) #
    volume_rotate = VolumeRotation(mode='bilinear') #'nearest'
    volume_rot = volume_rotate(volume.to(dtype=torch.float, device='cuda'),
                               R.to(dtype=torch.float, device='cuda'))

    #####################################################################
    
    p = pv.Plotter(point_smoothing = True, shape=(1, 3))
    fs = 15
    
    channel = volume[0,0,:,:,:].cpu().numpy()
    text = 'original'
    p.subplot(0, 0)
    p.add_text(text, position = 'upper_left', font_size = fs)
    p.add_volume(channel, cmap = "viridis_r", opacity = "linear")
    #p.add_axes()
    #print(channel.shape)

    

    ###########################################################
    channel = volume_rot[0,0,:,:,:].cpu().numpy()
    text = 'Random rotation'
    p.subplot(0, 1)
    p.add_text(text, position = 'upper_left', font_size = fs)
    p.add_volume(channel, cmap = "viridis_r", opacity = "linear")
    #p.add_axes()
    
    ###########################################################
    irot = 1
    channel = Rot90Seq(volume, iRot=irot)
    ch = channel[0,0,:,:,:].numpy()
    #print(type(ch))
    text = 'Rot90 around z(x-->y), RotIndx = '+str(irot)
    p.subplot(0, 2)
    p.add_text(text, position = 'upper_left', font_size = fs)
    p.add_volume(ch, cmap = "viridis_r", opacity = "linear")
    #p.add_axes()

    p.show()


