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


#pdb_ids = ["1pw2", "1ycr","2f6f"]#, "4f5t",
           ##"2am9", "3my5_a", "3w8m", "4ic8"] 

pdb_ids = ['1ycr']
path = "../data/"
path_list = [path+i+".pdb" for i in pdb_ids]

#print(path_list)

box_size = 60  #prog complains if box_size is float !!!!!! 
resolution = 1.0

volume = get_volume(path_list,
                       box_size,
                       resolution,
                       norm = False,
                       rot = False)

#print(volume.shape)
#print(torch.max(volume))

volume_sum = np.sum(volume.cpu().numpy(), axis=1)
#print(volume_sum.shape)
#print(volume_sum[0,:,:,:].shape)
#volume_sum = volume_sum[0,:,:,:]

grad = np.gradient(volume_sum[0,:,:,:])[0]
print(grad.min())
print(grad.max())


Agroup_names = ["Sulfur/Selenium"  , "Nitrogen Amide",
                "Nitrogen Aromatic", "Nitrogen Guanidinium",
                "Nitrogen Ammonium", "Oxygen Carbonyl",
                "Oxygen Hydroxyl"  , "Oxygen Carboxyl",
                "Carbon sp2"       , "Carbon Aromatic",
                "Carbon sp3"]


#idp = 0
#chan_id = 0 # Atomic group ids, range 0-10

#vol_low = volume[grad.clip(max=0),:,:,:]
#vol_high = volume[np.abs(grad),:,:,:]

vol_low = grad
#vol_low[ grad > 0] = 0

vol_high = grad
#vol_high[grad < 0] = 0

print(vol_low.min())
print(vol_high.min())


p = pv.Plotter(point_smoothing = True, shape=(1, 4))
#text = Agroup_names[chan_id]+" "+pdb_ids[idp]
p.subplot(0, 0)
p.add_volume(vol_high.clip(min=0, max=1), cmap = "viridis", opacity = "sigmoid")

p.subplot(0, 1)
#p.add_volume(np.abs(vol_low.clip(min=-1,max=0)), cmap = "viridis", opacity = "linear")
p.add_volume(np.abs(vol_low.clip(min=-1,max=0)), cmap = "viridis", opacity = "sigmoid")

p.subplot(0, 2)
p.add_volume(np.abs(grad), cmap = "viridis", opacity = "sigmoid")

p.subplot(0, 3)

p.add_volume(np.abs(grad), cmap = "viridis_r", opacity = "sigmoid")
#p.add_volume(np.abs(vol_low.clip(min=-1,max=0)), cmap = "viridis_r", opacity = "sigmoid")

p.show()

