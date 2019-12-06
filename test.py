import os
import sys
import torch
import torch.nn as nn
from src.cnn  import CnnModel
from src.volume import get_volume
from src.mapIO import write_map, greatest_dim
#from src.target import get_target
from src.util import grid2vec, unpad_map
import torch.optim as optim
import numpy as np

resolution = 1.000
kBT = 0.592 # T=298K, kB = 0.001987 kcal/(mol K)

#pdb_path = 'data/'
pdb_path = "/scratch/tr443/fragmap/data/"                                                          
pdb_ids = ["1ycr","1pw2", "2f6f", "4f5t", "1s4u", "2am9", "3my5_a", "3w8m","4ic8"]

map_names_list = ["apolar", "hbacc","hbdon", "meoo", "acec", "mamn"]
#map_path = 'data/maps/' 
map_path = "/scratch/tr443/fragmap/data/maps/"                                               

out_path = '/scratch/tr443/fragmap/output/'
#out_path = 'output/'

dim = greatest_dim(map_path, pdb_ids) + 1
box_size = int(dim*resolution)


batch_list = [pdb_path+pdb_ids[0]+".pdb"] 

#get volume tensor
norm = True
volume = get_volume(path_list = batch_list, 
                    box_size = box_size,
                    resolution = resolution,
                    norm = norm,
                    rotate = False)

#invoke model
torch.cuda.set_device(0)
model = CnnModel().cuda()
model.load_state_dict(torch.load(out_path+'net.pth'))
output = model(volume)

#criterion = nn.MSELoss()
#loss = criterion(output, target)


#save density maps to file
out_path = out_path + "maps/"
ori = [40.250, -8.472, 20.406] #!!!!!!!!!!!!!!!!!!!!!!!!

for imap in range(len(map_names_list)):
    
    out_name = pdb_ids[0]+"."+ map_names_list[imap]
    grid = output[0,imap,:,:,:].cpu().detach().numpy()
    #grid = unpad_map(grid, xpad = pad[0], ypad = pad[1], zpad = pad[2]) !!!!!!!!!!!!!!!!!!

    #convert from Free-E to density 
    #grid[grid <= 0.000] = 0.0001
    #vol = grid #-kBT *np.log(grid)  
   
    #if norm:   # inverse norm
    #    vol = grid*(gfe_max[i] - gfe_min[i]) + gfe_min[i] 
    #else:
    #vol = grid
        
    nx, ny, nz = grid.shape
    
    #flatten
    vec = grid2vec([nx,ny,nz], vol)

    write_map(vec, out_path, out_name, ori = ori,
              res = resolution, n = [nx,ny,nz])
    
