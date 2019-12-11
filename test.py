import os
import sys
import torch
import torch.nn as nn
from src.cnn  import CnnModel
from src.volume import get_volume
from src.mapIO import write_map, greatest_dim
from src.target import get_target
from src.util import grid2vec, unpad_mapc
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

params_file_name = 'net_params.pth'
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
                    rot = False)

#get testing map tensor
map_norm = True
test_map, pad, gfe_min, gfe_max, ori = get_target(map_path,
                                            map_names_list,
                                            pdb_ids = [pdb_ids[test_indx]],
                                            maxD = dim,
                                            kBT = kBT,
                                            density = False,
                                            map_norm = map_norm)

#convert target maps to torch.cuda
test_map = torch.from_numpy(test_map).float().cuda()


#-------------------------------------------------------------
#invoke model
torch.cuda.set_device(0)
model = CnnModel().cuda()
model.load_state_dict(torch.load(out_path + params_file_name))
model.eval() #Needed to set into inference mode
output = model(volume)


criterion = nn.MSELoss()
loss = criterion(output, test_map)
print("Testing Loss",loss.item())



for imap in range(len(map_names_list)):
   
    grid = output[0,imap,:,:,:].cpu().detach().numpy()
    grid = unpad_mapc(grid, pad = pad[0,:].astype(int))
    
    if map_norm == True:   # inverse norm
        vol = (1.0-grid)*(gfe_max[0,imap] - gfe_min[0,imap]) + gfe_min[0,imap]
    else:
        vol = grid
        
    nx, ny, nz = vol.shape              #get new dims       
    vec = grid2vec([nx,ny,nz], vol)     #flatten
    
    #write frag maps to output file
    out_name = pdb_ids[test_indx]+"."+ map_names_list[imap]
    write_map(vec, out_path, out_name, ori = ori[0,:],
              res = resolution, n = [nx,ny,nz])
