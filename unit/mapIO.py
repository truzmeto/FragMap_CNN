import numpy as np
import sys
import os
import pyvista as pv

sys.path.append(os.path.join(os.path.dirname(__file__), ".."))
from src.util import pad_map, vec2grid
from src.mapIO import read_map, write_map

frag_names = ["Benzene", "Propane", "H-bond Donor", "H-bond Acceptor"]

path_list = ["../data/maps/1ycr.benc.gfe.map",
             "../data/maps/1ycr.prpc.gfe.map",
             "../data/maps/1ycr.hbacc.gfe.map",
             "../data/maps/1ycr.hbdon.gfe.map"]

chan_id = 2 # range 0-3

######------------- Test the read_map -----------------#######
res, n_cells, dens = read_map(path_list[chan_id])
print("Extracted volume dimention --> ",dens.shape)
print("Specified dimension in the file header --> ", n_cells)

#plot map density
dens[dens > 0] = 0.0 #cutoff at zero!
channel = dens
p = pv.Plotter(point_smoothing = True)
p.add_volume(np.abs(channel), cmap = "viridis", opacity = "linear")
text = frag_names[chan_id]
p.add_text(text, position = 'upper_left', font_size = 16)
p.show()      


######------------- Testing write map ----------------#######
out_path = "./"
out_name = "test"
ori = [40.250, -8.472, 20.406]
res = 1.000
n = [5,5,5]
vec = 4*np.random.rand(n[0],n[1],n[2]) - 2.0
#write_map(vec, out_path, out_name, ori, res, n)!!!!!!!!!!!!!!!!!!!!!!!!!!!!
    
