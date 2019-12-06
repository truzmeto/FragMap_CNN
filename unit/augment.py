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
from src.augment import get_random_rotation


pdb_ids = ["1pw2", "1ycr","2f6f", "4f5t",
           "2am9", "3my5_a", "3w8m", "4ic8"] 

path = "../data/"
path_list = [path+i+".pdb" for i in pdb_ids]

box_size = 60  #prog complains if box_size is float !!!!!! 
resolution = 1.0
volume = get_volume(path_list,
                    box_size,
                    resolution,
                    norm = True,
                    rotate = True)

Agroup_names = ["Sulfur/Selenium"  , "Nitrogen Amide",
                "Nitrogen Aromatic", "Nitrogen Guanidinium",
                "Nitrogen Ammonium", "Oxygen Carbonyl",
                "Oxygen Hydroxyl"  , "Oxygen Carboxyl",
                "Carbon sp2"       , "Carbon Aromatic",
                "Carbon sp3"]

p = pv.Plotter(point_smoothing = True, shape=(2, 2))
fs = 15
idp = 1
chan_id = 8 # Atomic group ids, range 0-10

vol1 = volume[idp,chan_id,:,:,:].cpu().numpy()
text = Agroup_names[chan_id]+" "+pdb_ids[idp]
p.subplot(0, 0)
p.add_text(text, position = 'upper_left', font_size = fs)
p.add_volume(vol1, cmap = "viridis_r", opacity = "linear")
p.view_xy(negative=False)



rotated_vol = get_random_rotation(volume)
vol2 = rotated_vol[idp,chan_id,:,:,:].cpu().numpy()
text = "Rotated 1 " + Agroup_names[chan_id]+" "+pdb_ids[idp]
p.subplot(0, 1)
p.add_text(text, position = 'upper_left', font_size = fs)
p.add_volume(vol2, cmap = "viridis_r", opacity = "linear")
p.view_xy(negative=False)


rotated_vol_2 = get_random_rotation(volume)
vol3 = rotated_vol_2[idp,chan_id,:,:,:].cpu().numpy()
text = "Rotated 2 " + Agroup_names[chan_id]+" "+pdb_ids[idp]
p.subplot(1, 0)
p.add_text(text, position = 'upper_left', font_size = fs)
p.add_volume(vol3, cmap = "viridis_r", opacity = "linear")
p.view_xy(negative=False)


rotated_vol_3 = get_random_rotation(volume)
vol4 = rotated_vol_3[idp,chan_id,:,:,:].cpu().numpy()
text = "Rotated 3 " + Agroup_names[chan_id]+" "+pdb_ids[idp]
p.subplot(1, 1)
p.add_text(text, position = 'upper_left', font_size = fs)
p.add_volume(vol4, cmap = "viridis_r", opacity = "linear")
p.view_xy(negative=False)



p.show_axes()
p.link_views()
p.show()

