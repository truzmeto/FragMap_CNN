import os
import sys
import numpy as np

sys.path.append(os.path.join(os.path.dirname(__file__), ".."))
from src.volume import get_volume
from src.mapIO import greatest_dim
from src.target import get_target
from src.util import unpad_mapc

pdb_path = "/u1/home/tr443/data/fragData/"
pdb_ids = ["1ycr"]#, "1pw2", "2f6f"]#, "4f5t", "1s4u", "2am9", "3my5_a", "3w8m"]#,"4ic8"]
map_names_list = ["apolar"]#, "hbacc","hbdon", "meoo", "acec", "mamn"]
map_path = pdb_path  
dim = 100 #greatest_dim(map_path, pdb_ids) + 1

pdb_list = ["1ycr"]#, "1pw2"]    

target, pad, center = get_target(map_path,
                                 map_names_list,
                                 pdb_ids = pdb_list,
                                 maxD = dim)

print("Max dim = ", dim)
print('Padded Tensor Dims', target.shape)
print("Pads ", pad)
print("Max val", target.max())

