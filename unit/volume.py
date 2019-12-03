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


#A simple unit test!
pdb_ids = ["4f5t", "1ycr"]
path_list = ["../data/4f5t.pdb", "../data/1ycr.pdb"]
box_size = 60  #prog complains if box_size is float !!!!!! 
resolution = 1.0
volume = get_volume(path_list,
                    box_size,
                    resolution)

print(volume.size())
Agroup_names = ["Sulfur/Selenium"  , "Nitrogen Amide",
                "Nitrogen Aromatic", "Nitrogen Guanidinium",
                "Nitrogen Ammonium", "Oxygen Carbonyl",
                "Oxygen Hydroxyl"  , "Oxygen Carboxyl",
                "Carbon sp2"       , "Carbon Aromatic",
                "Carbon sp3"]

p = pv.Plotter(point_smoothing = True, shape=(1, 2))
fs = 15

chan_id = 1 # Atomic group ids, range 0-10
vol1 = volume[0,chan_id,:,:,:].cpu().numpy()
text = Agroup_names[chan_id]+" "+pdb_ids[0]
p.subplot(0, 0)
p.add_text(text, position = 'upper_left', font_size = fs)
p.add_volume(vol1, cmap = "viridis_r", opacity = "linear")

chan_id = chan_id 
vol2 = volume[1,chan_id,:,:,:].cpu().numpy()
text = Agroup_names[chan_id]+" "+pdb_ids[1]
p.subplot(0, 1)
p.add_text(text, position = 'upper_left', font_size = fs)
p.add_volume(vol2, cmap = "viridis_r", opacity = "linear")
p.show()

