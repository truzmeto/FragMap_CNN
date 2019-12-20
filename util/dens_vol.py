import os
import sys
import torch
from TorchProteinLibrary.Volume import TypedCoords2Volume
from TorchProteinLibrary.FullAtomModel import PDB2CoordsUnordered, Coords2TypedCoords
from TorchProteinLibrary.FullAtomModel import getRandomRotation, getRandomTranslation
from TorchProteinLibrary.FullAtomModel import CoordsRotate, CoordsTranslate, getBBox
import pyvista as pv
import numpy as np
sys.path.append(os.path.join(os.path.dirname(__file__), ".."))
from src.volume import get_volume

from scipy import ndimage



#pdb_ids = ["1pw2", "1ycr","2f6f"]#, "4f5t",
           ##"2am9", "3my5_a", "3w8m", "4ic8"] 

pdb_ids = ['1ycr']
#pdb_ids = ['4ic8']
path = "../data/"
path_list = [path+i+".pdb" for i in pdb_ids]

#print(path_list)

box_size = 100  #prog complains if box_size is float !!!!!! 
resolution = 1.0

volume, _ = get_volume(path_list,
                       box_size,
                       resolution,
                       norm = True,
                       rot = False)

#print(volume.shape)
#print(torch.max(volume))

volume_sum = np.sum(volume.cpu().numpy(), axis=1)
#print(volume_sum.shape)
#print(volume_sum[0,:,:,:].shape)
#volume_sum = volume_sum[0,:,:,:]

grad = np.gradient(volume_sum[0,:,:,:])[0]
print('grad1 min ',grad.min())
print('grad1 max ',grad.max())


Agroup_names = ["Sulfur/Selenium"  , "Nitrogen Amide",
                "Nitrogen Aromatic", "Nitrogen Guanidinium",
                "Nitrogen Ammonium", "Oxygen Carbonyl",
                "Oxygen Hydroxyl"  , "Oxygen Carboxyl",
                "Carbon sp2"       , "Carbon Aromatic",
                "Carbon sp3"]


vol_low = grad
#vol_low[ grad > 0] = 0

vol_high = grad
#vol_high[grad < 0] = 0

print('low vol min ',vol_low.min())
print('high vol min ',vol_high.min())


grad2 = np.gradient(grad[:,:,:])[0]
print('grad2 min ',grad2.min())
print('grad2 max ',grad2.max())

vol_low2 = grad2
#vol_low[ grad > 0] = 0

vol_high2 = grad2
#vol_high[grad < 0] = 0



laplace_test = ndimage.laplace(volume_sum)[0]
#print(grad2)
#print(laplace_test)


#p = pv.Plotter(point_smoothing = True, shape=(1, 6))
#fs = 15

#p.subplot(0, 0)
##p.add_volume(vol_high.clip(min=0, max=1), cmap = "viridis", opacity = "sigmoid")
##p.add_text('(+) first order gradient', position = 'upper_left', font_size = fs)


#p.subplot(0, 1)
##p.add_volume(np.abs(vol_low.clip(min=-1,max=0)), cmap = "viridis", opacity = "sigmoid")
##p.add_text('(-) first order gradient', position = 'upper_left', font_size = fs)


#p.subplot(0, 2)
##p.add_volume(np.abs(grad), cmap = "viridis", opacity = "sigmoid")
##p.add_text('Abs val first order gradient', position = 'upper_left', font_size = fs)


p = pv.Plotter(point_smoothing = True, shape=(1, 3))
fs=15
#color = 'viridis'
color = 'fire'

opac = 'sigmoid'
#opac = 'linear'


p.subplot(0, 0)
mi=-0.1
mx=1

p.add_volume(vol_high2.clip(min=mi, max=mx), cmap = color, opacity = opac)
p.add_text('(-0.25 to +1) second order gradient', position = 'upper_left', font_size = fs)

p.subplot(0, 1)
mi=-0.1
mx=1

p.add_volume(laplace_test.clip(min=mi, max=mx), cmap = color, opacity = opac)
p.add_text('CAVITY (-0.25 to +1) laplace 2nd deriv', position = 'upper_left', font_size = fs)

p.subplot(0, 2)
mi=-1
mx=0

p.add_volume(np.abs(laplace_test.clip(min=mi, max=mx)), cmap = color, opacity = opac)
p.add_text('DENSE abs(-1 to 0.5) laplace 2nd deriv', position = 'upper_left', font_size = fs)

p.show()

