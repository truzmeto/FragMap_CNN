import os
import sys
import torch
import pyvista as pv
import numpy as np

sys.path.append(os.path.join(os.path.dirname(__file__), ".."))
from src.volume import get_volume


pdb_ids = ["3my5_a", "4f5t"]#, "4wj9", "3bi0", "1d6e_nopep_fixed",
           #"1ycr", "1pw2", "2f6f", "4ic8", "1s4u", "2am9",
           #"1bvi_m1_fixed", "4djw", "4lnd_ba1", "4obo",
           #"1h1q", "3fly", "4gih", "2gmx", "4hw3", "4ypw_prot_nocter",
           # "3w8m", "2qbs", "4jv7", "5q0i", "1r2b", "2jjc"]


#print(len(pdb_ids))
ipdb = len(pdb_ids)-2
path = "/u1/home/tr443/data/fragData/"

box_size = 90  #prog complains if box_size is float !!!!!!
resolution = 1.0

for i in range(len(pdb_ids)):

    #    #path_list = [path+i+".pdb" for i in [pdb_ids[ipdb]]]
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
chan_id = 1 # Atomic group ids, range 0-10
vol1 = volume[0,chan_id,:,:,:].cpu().numpy()
text = Agroup_names[chan_id]+" "+pdb_ids[idp]

p.subplot(0, 0)
p.add_text(text, position = 'upper_left', font_size = fs)
p.add_volume(vol1, cmap = "viridis_r", opacity = "linear")

chan_id = chan_id+1
idp = idp
vol2 = volume[0,chan_id,:,:,:].cpu().numpy()
text = Agroup_names[chan_id]+" "+pdb_ids[idp]
p.subplot(0, 1)
p.add_text(text, position = 'upper_left', font_size = fs)
p.add_volume(vol2, cmap = "viridis_r", opacity = "linear")
p.show()

