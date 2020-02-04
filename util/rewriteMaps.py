import numpy as np
import sys
import os
sys.path.append(os.path.join(os.path.dirname(__file__), ".."))

from src.mapIO import read_map, write_map
from src.util import box_face_ave, box_face_med
import matplotlib.pyplot as plt


frag_names = ["apolar", "hbacc", "hbdon", "meoo","acec", "mamn"]
#path = "../data/maps/"
path = "/u1/home/tr443/data/fragData/maps/"
pdb_ids = ["1ycr", "1pw2", "2f6f", "4f5t", "1s4u", "2am9", "3my5_a", "3w8m","4ic8"]
pdb_id = pdb_ids[0]
tail = ".gfe.map"
path_list = [path+pdb_id+"."+name+tail for name in frag_names]



for i in range(len(path_list)):
    res, _, gfe, center = read_map(path_list[i])

    bsl1 = box_face_med(gfe)
    bsl2 = np.median(gfe)
    print(bsl1, bsl2)
    nx, ny, nz = gfe.shape
    new_shape = nx*ny*nz    
    vec = np.reshape(gfe,new_shape, order="F") - bsl1
    
    out_name = pdb_id + "." + frag_names[i]
    #write_map(vec, out_path+"maps/", out_name, center[0,:],
    #          res = resolution, n = [nx,ny,nz])
    #print(center)
    write_map(vec, "", out_name, center, res = res, n = [nx,ny,nz])

 
