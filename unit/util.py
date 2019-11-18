import numpy as np
import sys
import os
import pyvista as pv
sys.path.append(os.path.join(os.path.dirname(__file__), ".."))
from src.mapIO import read_map 
from src.util import pad_map, unpad_map 

frag_names = ["Benzene", "Propane", "H-bond Donor", "H-bond Acceptor"]
path_list = ["../data/maps/1ycr.benc.gfe.map",
             "../data/maps/1ycr.prpc.gfe.map",
             "../data/maps/1ycr.hbacc.gfe.map",
             "../data/maps/1ycr.hbdon.gfe.map"]

chan_id = 0 # range 0-3
_, _, dens = read_map(path_list[chan_id])


######------------- Test padding ----------------#######
pad_dens, xpad, ypad, zpad = pad_map(dens)

if np.abs(pad_dens.sum() - dens.sum()) > 0.000001:
    print("Error! Zero padding should not affect the sum")
    print("Padded sum = ", pad_dens.sum())
    print("Map sum = ", dens.sum())
else:
    print("Padding test passed!")
    
    
######------------- Test unpadding ----------------#######
ori_dim = dens.shape
up_dens = unpad_map(pad_dens, xpad, ypad, zpad)
unpad_dim = up_dens.shape
i = 0; dp = 0
for item in ori_dim:
    dp = item - unpad_dim[i]
    i+=1
    if dp != 0:
        print("Original dim is not same as padded dim!")
        break
    
    
#compare original and unpadded map volumes by visialization
p = pv.Plotter(point_smoothing = True, shape=(1, 2))
fs = 16

p.subplot(0, 0)
text = frag_names[chan_id]+" original " + "dim = "+ str(ori_dim)
p.add_text(text, position = 'upper_left', font_size = fs)
p.add_volume(np.abs(dens), cmap = "viridis", opacity = "linear")

p.subplot(0, 1)
text = frag_names[chan_id]+"un-padded" + "dim = "+ str(unpad_dim)
p.add_text(text, position = "upper_left", font_size = fs)
p.add_volume(np.abs(up_dens), cmap = "viridis", opacity = "linear")


p.show()
