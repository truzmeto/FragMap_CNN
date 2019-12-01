import torch
from TorchProteinLibrary.Volume import VolumeRotation
from TorchProteinLibrary.FullAtomModel import getRandomRotation, getRandomTranslation
import pyvista as pv
import numpy as np

volume = torch.zeros(1,1,35,35,35)
volume[:,:,5:25,5,5] = 0.8

R = getRandomRotation(1) #
volume_rotate = VolumeRotation(mode='bilinear')
volume_rot = volume_rotate(volume.to(dtype=torch.float, device='cuda'),
                           R.to(dtype=torch.float, device='cuda'))

#####################################################################

p = pv.Plotter(point_smoothing = True, shape=(1, 2))
fs = 15

channel = volume[0,0,:,:,:].cpu().numpy()
text = 'original'
p.subplot(0, 0)
p.add_text(text, position = 'upper_left', font_size = fs)
p.add_volume(channel, cmap = "viridis_r", opacity = "linear")

channel = volume_rot[0,0,:,:,:].cpu().numpy()
text = 'rotated'
p.subplot(0, 1)
p.add_text(text, position = 'upper_left', font_size = fs)
p.add_volume(channel, cmap = "viridis_r", opacity = "linear")

p.show()
