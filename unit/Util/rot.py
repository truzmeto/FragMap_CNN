import torch
from TorchProteinLibrary.Volume import VolumeRotation
from TorchProteinLibrary.FullAtomModel import getRandomRotation, getRandomTranslation
import pyvista as pv
import numpy as np
import sys
import os
from scipy import ndimage
from Shapes3D import get3D_rod

sys.path.append(os.path.join(os.path.dirname(__file__), "../.."))
from src.Util.rot24 import Rot90Seq


def rotate_ligand(ligand, rotation_angle):    
   
    ligand = ndimage.interpolation.rotate(ligand,
                                          angle = rotation_angle,
                                          axes=(2,0),
                                          reshape=False,
                                          order=0,
                                          mode= 'nearest', #'constant',
                                          cval=0.0)
   
    return ligand



if __name__=='__main__':
   
    volume =  get3D_rod()
    R = getRandomRotation(1) #
    volume_rotate = VolumeRotation(mode='bilinear') #'nearect'
    volume_rot = volume_rotate(volume.to(dtype=torch.float, device='cuda'),
                               R.to(dtype=torch.float, device='cuda'))

    #####################################################################
    
    p = pv.Plotter(point_smoothing = True, shape=(1, 4))
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
    text = 'RandomRot TPL'
    p.subplot(0, 2)
    p.add_text(text, position = 'upper_left', font_size = fs)
    p.add_volume(channel, cmap = "viridis_r", opacity = "linear")
    #p.add_axes()
    
    ###########################################################
    irot = 13
    channel = Rot90Seq(volume, iRot=irot)
    ch = channel[0,0,:,:,:].numpy()
    #print(type(ch))
    text = 'Rot90 around z(x-->y) \n RotIndx = '+str(irot)
    p.subplot(0, 1)
    p.add_text(text, position = 'upper_left', font_size = fs)
    p.add_volume(ch, cmap = "viridis_r", opacity = "linear")
    #p.add_axes()


    
    ###########################################################
    ch1 = rotate_ligand(volume[0,0,:,:,:].numpy(), rotation_angle=20)    
    #print(type(ch))
    text = 'scipy.ndRot \n  0 order interpol'
    p.subplot(0, 3)
    p.add_text(text, position = 'upper_left', font_size = fs)
    p.add_volume(ch1, cmap = "viridis_r", opacity = "linear")
    #p.add_axes()


    
    p.show()


