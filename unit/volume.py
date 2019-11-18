import os
import sys
import torch
from TorchProteinLibrary.Volume import TypedCoords2Volume
from TorchProteinLibrary.FullAtomModel import PDB2CoordsUnordered, Coords2TypedCoords
from TorchProteinLibrary.FullAtomModel import getRandomRotation, getRandomTranslation
from TorchProteinLibrary.FullAtomModel import CoordsRotate, CoordsTranslate, getBBox
import pyvista as pv
sys.path.append(os.path.join(os.path.dirname(__file__), ".."))
from src.volume import get_volume


#A simple unit test!
path_list = ["../data/1ycr.pdb"]
box_size = 58  # prog complains if box_size is float !!!!!! 
resolution = 1.0
volume = get_volume(path_list, box_size, resolution)
print(volume.shape)

Agroup_names = ["Sulfur/Selenium", "Nitrogen Amide",
                "Nitrogen Aromatic", "Nitrogen Guanidinium",
                "Nitrogen Ammonium", "Oxygen Carbonyl",
                "Oxygen Hydroxyl", "Oxygen Carboxyl",
                "Carbon sp2","Carbon Aromatic",
                "Carbon sp3"]


p = pv.Plotter(point_smoothing = True, shape=(1, 2))
fs = 15

chan_id = 1 # Atomic group ids, range 0-10
channel = volume[0,chan_id,:,:,:].cpu().numpy()
text = Agroup_names[chan_id]
p.subplot(0, 0)
p.add_text(text, position='upper_left', font_size=fs)
p.add_volume(channel, cmap = "viridis_r", opacity = "linear")

chan_id = 10 
channel = volume[0,chan_id,:,:,:].cpu().numpy()
text = Agroup_names[chan_id]
p.subplot(0, 1)
p.add_text(text, position='upper_left', font_size=fs)
p.add_volume(channel, cmap = "viridis_r", opacity = "linear")

p.show()

