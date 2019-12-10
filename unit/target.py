import os
import sys
import numpy as np

sys.path.append(os.path.join(os.path.dirname(__file__), ".."))
from src.volume import get_volume
from src.mapIO import greatest_dim
from src.target import get_target
from src.util import unpad_map

map_norm = False
kBT = 0.592 # T=298K, kB = 0.001987 kcal/(mol K)

pdb_path = '../data/'
pdb_ids = ["1ycr", "1pw2", "2f6f", "4f5t", "1s4u"]#, "2am9", "3my5_a", "3w8m"]#,"4ic8"]

map_names_list = ["apolar", "hbacc","hbdon"]#, "meoo", "acec", "mamn"]
map_path = '../data/maps/' 

dim = 70 #greatest_dim(map_path, pdb_ids) + 1

pdb_list = ["1ycr"]#, "1pw2"]    
#get target map tensor
target, pad, gfe_min, gfe_max = get_target(map_path,
                                           map_names_list,
                                           pdb_ids = pdb_list,
                                           maxD = dim,
                                           kBT = kBT,
                                           cutoff = False,
                                           density = False,
                                           map_norm = map_norm)

print("Max dim = ", dim)
print('Padded Tensor Dims', target.shape)
print("Pads ", pad)

#unpad 1st structure and 1st volume
up_gfe = unpad_map(target[0,0,:,:,:], pad[0][0]+1,pad[0][1]+1, pad[0][2]+1)
print("Unpadded 1st map", up_gfe.shape)
