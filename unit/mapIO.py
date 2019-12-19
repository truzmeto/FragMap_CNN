import numpy as np
import sys
import os
import pyvista as pv

sys.path.append(os.path.join(os.path.dirname(__file__), ".."))
from src.util import pad_mapc, vec2grid
from src.mapIO import read_map, write_map


pdb_ids = ["1ycr","1pw2","2f6f", "4f5t", "2am9", "3my5_a", "3w8m", "4ic8"] 

path = "../data/maps/"
frag_names = ["apolar", "hbacc","hbdon", "meoo", "acec", "mamn"]
idx = 7 # pdb id
pdb_id = pdb_ids[idx]
path_list = [path+pdb_id+"." + i + ".gfe.map" for i in frag_names]

chan_id = 5 #map id

######------------- Test the read_map -----------------#######
res, n_cells, dens, center = read_map(path_list[chan_id])
print("Extracted volume dimention --> ",dens.shape)
print("Specified dimension in the file header --> ", n_cells)

#plot map density
#dens[dens < 0] = 0.0 #cutoff at zero!
channel = np.exp(-dens[:,:,33:35]/0.58)
p = pv.Plotter(point_smoothing = True)
p.add_volume(channel, cmap = "viridis", opacity = "linear")
text =  pdb_id + "." + frag_names[chan_id] + " Density"#+"  dim = " + str(dens.shape) 
p.add_text(text, position = 'upper_left', font_size = 16)
p.show()      




######------------- Testing write map ----------------#######
out_path = "./"
out_name = "test"
ori = center#[40.250, -8.472, 20.406]
res = 1.000
n = [5,5,5]
vec = 4*np.random.rand(n[0],n[1],n[2]) - 2.0
#write_map(vec, out_path, out_name, ori, res, n)!!!!!!!!!!!!!!!!!!!!!!!!!!!!
    
