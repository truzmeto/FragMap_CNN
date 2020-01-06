import numpy as np
import sys
import os
import pyvista as pv
sys.path.append(os.path.join(os.path.dirname(__file__), ".."))
from src.mapIO import read_map , greatest_dim
from src.util import pad_mapc, unpad_mapc 
import matplotlib.pyplot as plt

frag_names = ["Benzene", "Propane", "H-bond Donor", "H-bond Acceptor"]
frag_names_short = ["benc", "prpc", "hbacc", "hbdon"]
path = "../data/maps/"
pdb_id = "1ycr"
tail = ".gfe.map"
path_list = [path+pdb_id+"."+name+tail for name in frag_names_short]

chan_id = 1 # range 0-3
_, _, gfe, _ = read_map(path_list[chan_id])


maxD = greatest_dim("../data/maps/", ["1ycr", "1pw2", "2f6f"])
print(maxD)

######------------- Test padding ----------------#######
pad_gfe, pad = pad_mapc(gfe, maxD, pad_val = 0.0)
print('Original map',gfe.shape)
print('Padded map',pad_gfe.shape)


if np.abs(pad_gfe.sum() - gfe.sum()) > 0.000001:
    print("Error! Zero padding should not affect the sum")
    print("Padded sum = ", pad_gfe.sum())
    print("Map sum = ", gfe.sum())
else:
    print("Padding test passed!")
    
    
######------------- Test unpadding ----------------#######
ori_dim = gfe.shape
up_gfe = unpad_mapc(pad_gfe, pad)
unpad_dim = up_gfe.shape
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
text = frag_names[chan_id]+" original " + "dim = "+ str(pad_gfe.shape)
p.add_text(text, position = 'upper_left', font_size = fs)
p.add_volume(np.abs(gfe), cmap = "viridis", opacity = "linear")

p.subplot(0, 1)
text = frag_names[chan_id]+" unpadded " + "dim = "+ str(unpad_dim)
p.add_text(text, position = "upper_left", font_size = fs)
p.add_volume(np.abs(up_gfe), cmap = "viridis", opacity = "linear")
p.show()

