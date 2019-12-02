import torch
from TorchProteinLibrary.Volume import VolumeRotation
from TorchProteinLibrary.FullAtomModel import getRandomRotation, getRandomTranslation
import pyvista as pv
import numpy as np

volume = torch.zeros(1,1,55,54,53)
volume[:,:,27,26,10:30] = 0.5
volume[:,:,27,26,30:31] = 1.0

volume[:,:,27:37,26,10] = 0.5
volume[:,:,37:38,26,10] = 1.

volume[:,:,27,26:36,10] = 0.5
volume[:,:,27,36:37,10] = 1.

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
#p.add_axes()

print(channel.shape)
channel = volume_rot[0,0,:,:,:].cpu().numpy()
#channel = np.transpose(channel, axes=[2,0,1])
print(channel.shape)
text = 'rotated'
p.subplot(0, 1)
p.add_text(text, position = 'upper_left', font_size = fs)
p.add_volume(channel, cmap = "viridis_r", opacity = "linear")
p.add_axes()
p.show()
