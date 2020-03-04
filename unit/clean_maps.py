import sys
import os
import numpy as np
import torch

sys.path.append(os.path.join(os.path.dirname(__file__), ".."))
from src.mapIO import  write_map
from src.target import get_target
from src.util import pad_mapc, unpad_mapc


pdb_ids = ["1bvi_m1_fixed"]#"1ycr", "1pw2", "1s4u", "2f6f"]#, "2am9", "3my5_a", "3w8m", "4ic8", "4f5t"]
path = "/u1/home/tr443/data/fragData/"
#path = "../../data/"

dim = int(53)
map_names_list = ["apolar", "hbacc","hbdon", "meoo", "acec", "mamn"]

gfe, pad, center = get_target(path,
                              map_names_list,
                              pdb_ids = pdb_ids,
                              maxD = dim)

gfe = gfe.cpu().detach().numpy()
for i in range(6):
    
    grid = gfe[0,i,:,:,:]#.numpy()
    grid = unpad_mapc(grid, pad = pad[0,:])
    
    nx, ny, nz = grid.shape
    new_shape = nx*ny*nz
    vec = np.reshape(grid, new_shape, order="F")
    
    out_name = pdb_ids[0] + "." + map_names_list[i]+"P"
    write_map(vec, "../output/maps/", out_name, center[0,:], res = 1.0, n = [nx,ny,nz])

