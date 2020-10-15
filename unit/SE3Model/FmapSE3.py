"""
Works with old repo
"""
import os
import sys
import torch
import pyvista as pv
import numpy as np

sys.path.append(os.path.join(os.path.dirname(__file__), "../.."))
from src.Util.volume import get_volume
from src.SE3Model.convSE import FragMapSE3


pdb_ids = ["1ycr"]#,"1pw2"]
path = "/u1/home/tr443/data/fragData/"
path_list = [path+i+".pdb" for i in pdb_ids]

box_size = 60
resolution = 1.0

inp, _ = get_volume(path_list,
                    box_size,
                    resolution,
                    norm = True,
                    rot = False,
                    trans = False)

# =============================================================================
#
model = FragMapSE3().cuda()
out = model(inp)

#compare input and output shapes
print("11 input channels VS 4 output channels")
print("Input dimension -->", inp.size())
print("Output dimension -->",out.size())

Agroup_names = ["Sulfur/Selenium"  , "Nitrogen Amide",
                "Nitrogen Aromatic", "Nitrogen Guanidinium",
                "Nitrogen Ammonium", "Oxygen Carbonyl",
                "Oxygen Hydroxyl"  , "Oxygen Carboxyl",
                "Carbon sp2"       , "Carbon Aromatic",
                "Carbon sp3"]

p = pv.Plotter(point_smoothing = True, shape=(1, 2))


fs = 15 ; idp = 0
chan_id = 1 # Atomic group ids, range 0-10
inp1 = inp[idp,chan_id,:,:,:].detach().cpu().numpy()
text = Agroup_names[chan_id]+" "+pdb_ids[idp]
p.subplot(0, 0)
p.add_text(text, position = 'upper_left', font_size = fs)
p.add_volume(inp1, cmap = "viridis_r", opacity = "linear")

chan_id = 5
idp = idp
out1 = out[idp,chan_id,:,:,:].detach().cpu().numpy()
text = "output"#Agroup_names[chan_id]+" "+pdb_ids[idp]
p.subplot(0, 1)
p.add_text(text, position = 'upper_left', font_size = fs)
p.add_volume(out1, cmap = "viridis_r", opacity = "linear")
p.show()

