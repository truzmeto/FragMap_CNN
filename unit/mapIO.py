import numpy as np
import sys
import os
import pyvista as pv
sys.path.append(os.path.join(os.path.dirname(__file__), ".."))
from src.util import pad_mapc, vec2grid, get_bin_frequency
from src.mapIO import read_map, write_map
from src.volume import get_volume


pdb_ids =["1ycr","1pw2", "2f6f", "4f5t", "1s4u", "2am9", "3my5_a", "3w8m", "4ic8"]

Agroup_names = ["Sulfur/Selenium"  , "Nitrogen Amide",
                "Nitrogen Aromatic", "Nitrogen Guanidinium",
                "Nitrogen Ammonium", "Oxygen Carbonyl",
                "Oxygen Hydroxyl"  , "Oxygen Carboxyl",
                "Carbon sp2"       , "Carbon Aromatic",
                "Carbon sp3"]
path = "data/maps/"
frag_names = ["apolar", "hbacc","hbdon", "meoo", "acec", "mamn"]
idx = 8 # pdb id
pdb_id = pdb_ids[idx]
path_list = [path+pdb_id+"." + i + ".gfe.map" for i in frag_names]
#pdb_ids = ['4ic8']

chan_id = 0
######------------- Test the read_map -----------------#######
res, n_cells, dens, center = read_map(path_list[chan_id])
print("Extracted volume dimention --> ",dens.shape)
print("Specified dimension in the file header --> ", n_cells)
temp = np.copy(dens)
temp2 = np.copy(dens)
temp3 = np.copy(dens)

#plot map density
dens[dens <4] = 0
# dens[dens > -1] = 0
p = pv.Plotter(point_smoothing = True, shape=(2, 3))
p.subplot(0,0)
p.add_volume(dens[n_cells[0]//2:n_cells[0]//2+5,:,:], cmap = "viridis", opacity = "linear")
text =  pdb_id + "." + frag_names[chan_id] +" GFE > 4"
p.add_text(text, position = 'upper_left', font_size = 16)
# p.show()      

temp[temp <3] = 0
temp[temp > 4] = 0
p.subplot(0,1)
p.add_volume(temp[n_cells[0]//2:n_cells[0]//2+5,:,:], cmap = "viridis", opacity = "linear")
text =  pdb_id + "." + frag_names[chan_id] +" 3 < GFE < 4"
p.add_text(text, position = 'upper_left', font_size = 16)

temp2[temp2 <2] = 0
temp2[temp2 > 3] = 0
p.subplot(0,2)
p.add_volume(temp2[n_cells[0]//2:n_cells[0]//2+5,:,:], cmap = "viridis", opacity = "linear")

text =  pdb_id + "." + frag_names[chan_id] +" 2 < GFE < 3"
p.add_text(text, position = 'upper_left', font_size = 16)

temp3[temp3 <1.5] = 0
temp3[temp3 > 2.0] = 0
p.subplot(1,0)
p.add_volume(temp3[n_cells[0]//2:n_cells[0]//2+5,:,:], cmap = "viridis", opacity = "linear")

text =  pdb_id + "." + frag_names[chan_id] +" 1.5 < GFE < 2"
p.add_text(text, position = 'upper_left', font_size = 16)

pdb_ids = ['4ic8']
path = "data/"
path_list = [path+i+".pdb" for i in pdb_ids]
#print(path_list)
box_size = 103  #prog complains if box_size is float !!!!!! 
resolution = 1.0
volume, _ = get_volume(path_list,
                       box_size,
                       resolution,
                       norm = True)

volume_sum= np.sum(volume.cpu().numpy(), axis=1)[0]
# vol2 = volume[idx,chan_id,:,:,:].cpu().numpy()
text = "Summed volume map"
p.subplot(1, 1)
p.add_text(text, position = 'upper_left', font_size = 15)
p.add_volume(volume_sum[n_cells[0]//2:n_cells[0]//2+5,:,:] ,cmap = "viridis_r", opacity = "linear")
p.link_views()
p.show()



######------------- Testing write map ----------------#######
out_path = "./"
out_name = "test"
ori = center#[40.250, -8.472, 20.406]
res = 1.000
n = [5,5,5]
vec = 4*np.random.rand(n[0],n[1],n[2]) - 2.0
#write_map(vec, out_path, out_name, ori, res, n)!!!!!!!!!!!!!!!!!!!!!!!!!!!!
    
