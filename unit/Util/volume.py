import os
import sys
import torch
import pyvista as pv
import numpy as np

sys.path.append(os.path.join(os.path.dirname(__file__), "../.."))
from src.Util.volume import get_volume

pdb_ids = ["3my5_a"]#, "4f5t", "4wj9"]
ipdb = len(pdb_ids)-2
path = "/u1/home/tr443/data/fragData/"
box_size = 90  #prog complains if box_size is float !!!!!
resolution = 1.0

for i in range(len(pdb_ids)):

    #path_list = [path+i+".pdb" for i in [pdb_ids[ipdb]]]
    path_list = [path + pdb_ids[i] + ".pdb"]
    print(pdb_ids[i])
    volume, _ = get_volume(path_list,
                           box_size,
                           resolution,
                           norm = True,
                           rot = False,
                           trans = False)

print(volume.shape)

Agroup_names = ["Sulfur/Selenium"  , "Nitrogen Amide",
                "Nitrogen Aromatic", "Nitrogen Guanidinium",
                "Nitrogen Ammonium", "Oxygen Carbonyl",
                "Oxygen Hydroxyl"  , "Oxygen Carboxyl",
                "Carbon sp2"       , "Carbon Aromatic",
                "Carbon sp3"]

p = pv.Plotter(point_smoothing = True, shape=(1, 2))
fs = 15
idp = ipdb
chan_id = 0 # Atomic group ids, range 0-10
vol1 = volume[0,chan_id,:,:,:].cpu().numpy()
text = Agroup_names[chan_id]+" "+pdb_ids[idp]

p.subplot(0, 0)
p.add_text(text, position = 'upper_left', font_size = fs)
p.add_volume(vol1, cmap = "viridis_r", opacity = "linear")

chan_id = chan_id + 1
idp = idp
vol2 = volume[0,chan_id,:,:,:].cpu().numpy()
text = Agroup_names[chan_id]+" "+pdb_ids[idp]
p.subplot(0, 1)
p.add_text(text, position = 'upper_left', font_size = fs)
p.add_volume(vol2, cmap = "viridis_r", opacity = "linear")
p.show()

