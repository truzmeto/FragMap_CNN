import os
import sys
import torch
import torch.nn as nn
from src.cnn  import CnnModel
from src.volume import get_volume
from src.mapIO import write_map, greatest_dim
from src.target import get_target
from src.util import grid2vec, unpad_map
import torch.optim as optim
import numpy as np

resolution = 1.000
kBT = 0.592 # T=298K, kB = 0.001987 kcal/(mol K)

#pdb_path = 'data/'
pdb_path = "/scratch/tr443/fragmap/data/"                                                          
pdb_ids = ["1ycr","1pw2", "2f6f", "4f5t", "1s4u", "2am9", "3my5_a", "3w8m", "4ic8"]

map_names_list = ["apolar", "hbacc","hbdon", "meoo", "acec", "mamn"]
#map_path = 'data/maps/' 
map_path = "/scratch/tr443/fragmap/data/maps/"                                               

out_path = '/scratch/tr443/fragmap/output/'
#out_path = 'output/'

test_file_name = 'net_params1.pth'
test_indx = 8

dim = greatest_dim(map_path, pdb_ids) + 1
box_size = int(dim*resolution)


batch_list = [pdb_path+pdb_ids[test_indx]+".pdb"] 
#get volume tensor
norm = True
volume = get_volume(path_list = batch_list, 
                    box_size = box_size,
                    resolution = resolution,
                    norm = norm,
                    rotate = False)

#get testing map tensor
map_norm = True
test_map, pad, gfe_min, gfe_max = get_target(map_path,
                                             map_names_list,
                                             pdb_ids = [pdb_ids[test_indx]], #? the last one for now
                                             maxD = dim,
                                             kBT = kBT,
                                             cutoff = False,
                                             density = False,
                                             map_norm = map_norm)

#convert target maps to torch.cuda
test_map = torch.from_numpy(test_map).float().cuda()


#-------------------------------------------------------------------
#invoke model
torch.cuda.set_device(0)
model = CnnModel().cuda()
model.load_state_dict(torch.load(out_path + test_file_name))
output = model(volume)


criterion = nn.MSELoss()
loss = criterion(output, test_map)
print("Testing Loss",loss.item())


#save density maps to file
ori = [23.699, -20.418, 14.198] 

for imap in range(len(map_names_list)):
   
    grid = output[0,imap,:,:,:].cpu().detach().numpy()
    grid = unpad_map(grid, xpad = pad[0][0], ypad = pad[0][1], zpad = pad[0][2])

    #convert from Free-E to density 
    #grid[grid <= 0.000] = 0.0001
    #vol = grid #-kBT *np.log(grid)  

    
    if map_norm:   # inverse norm
        grid = grid*(gfe_max[:,imap] - gfe_min[:,imap]) + gfe_min[:,imap] 
 
       
    nx, ny, nz = grid.shape              #get new dims       
    vec = grid2vec([nx,ny,nz], grid)     #flatten
    
    #write frag maps to output file
    out_name = pdb_ids[test_indx]+"."+ map_names_list[imap]
    write_map(vec, out_path, out_name, ori = ori,
              res = resolution, n = [nx,ny,nz])
    
