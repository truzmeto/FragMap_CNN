import torch
from TorchProteinLibrary.Volume import VolumeRotation
from TorchProteinLibrary.FullAtomModel import getRandomRotation, getRandomTranslation
import pyvista as pv
import numpy as np

volume = torch.zeros(1,1,55,54,53)
length = 15
st = [27,26,25]

volume[:,:,st[0]:st[0]+length,st[1],st[2]] = 0.5
volume[:,:,st[0]+length:st[0]+length+1,st[1],st[2]] = 0.0

volume[:,:,st[0],st[1]:st[1]+length,st[2]] = 0.5
volume[:,:,st[0],st[1]+length:st[1]+length+1,st[2]] = 1.

volume[:,:,st[0],st[1],st[2]:st[2]+length] = 0.5
volume[:,:,st[0],st[1],st[2]+length:st[2]+length+1] = 2.0


R = getRandomRotation(1) #
volume_rotate = VolumeRotation(mode='bilinear') #'nearest'
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


###########################################################
#channel = volume_rot[0,0,:,:,:].cpu().numpy()
channel = volume[0,0,:,:,:].cpu()#.numpy()
#channel = np.transpose(channel, axes=[2,0,1])
channel = channel.transpose(0, 1).flip(2).numpy()
#channel = channel.flip(2).numpy()

print(channel.shape)
text = 'rotated'
p.subplot(0, 1)
p.add_text(text, position = 'upper_left', font_size = fs)
p.add_volume(channel, cmap = "viridis_r", opacity = "linear")
p.add_axes()
p.show()
