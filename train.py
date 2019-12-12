import os
import sys
import torch
import torch.nn as nn
from src.cnn  import CnnModel
from src.volume import get_volume
from src.mapIO import greatest_dim, write_map
from src.target import get_target
from src.util import grid2vec, sample_batch, unpad_mapc
import torch.optim as optim
import numpy as np


#model params
lrt = 0.0001
#lrd = 0.0001
wd = 0.00001
max_epoch = 2000
batch_size = 2 #number of structures in a batch

norm = True
map_norm = True
nsample = 24

#physical params
resolution = 1.000
kBT = 0.592 # T=298K, kB = 0.001987 kcal/(mol K)

#pdb_path = 'data/'
pdb_path = "/scratch/tr443/fragmap/data/"                                                          
pdb_ids = ["1ycr", "1pw2", "2f6f", "4f5t", "1s4u", "2am9", "3my5_a", "3w8m"]#,"4ic8"]

map_names_list = ["apolar", "hbacc","hbdon", "meoo", "acec", "mamn"]
#map_path = 'data/maps/' 
map_path = "/scratch/tr443/fragmap/data/maps/"                                               

out_path = '/scratch/tr443/fragmap/output/'
#out_path = 'output/'


dim = greatest_dim(map_path, pdb_ids) + 1
box_size = int(dim*resolution)
params_file_name = 'net_params.pth'

#invoke model
torch.cuda.set_device(0)
model = CnnModel().cuda()
criterion = nn.MSELoss()
#criterion = nn.L1Loss()
optimizer = optim.Adam(model.parameters(), lr = lrt, weight_decay = wd )
#optimizer = optim.SGD(model.parameters(), lr = lrt, momentum = 0.9)

rand_rotations = True

for batches in range(nsample):

    #sample batch list from all structures
    batch_list, pdb_list = sample_batch(batch_size,
                                        pdb_ids,
                                        pdb_path,
                                        shuffle = True)
    #get batch volume tensor
    volume, rot_matrix = get_volume(path_list = batch_list, 
                                box_size = box_size,
                                resolution = resolution,
                                norm = norm,
                                rot = rand_rotations,
                                trans = False)
    
    #get target map tensor
    target, pad, gfe_min, gfe_max, center = get_target(map_path,
                                                map_names_list,
                                                pdb_ids = pdb_list,
                                                maxD = dim,
                                                kBT = kBT,
                                                density = False,
                                                map_norm = map_norm)
    
    #convert target maps to torch.cuda
    target = torch.from_numpy(target).float().cuda()

    if rand_rotations:
        target = grid_rot(target, batch_size, rot_matrix)
        
    
    #perform forward and backward iterations
    for epoch in range(max_epoch):
        
        optimizer.zero_grad()
        output = model(volume)
        loss = criterion(output, target)
        loss.backward()
        optimizer.step()
        
        if epoch % 40 == 0:
            print('{0},{1},{2}'.format(batches, epoch, loss.item()))
            
#save trained parameters        
torch.save(model.state_dict(), out_path+params_file_name)


#for imap in range(len(map_names_list)):
   
#    grid = output[0,imap,:,:,:].cpu().detach().numpy()
#    grid = unpad_mapc(grid, pad = pad[0,:].astype(int))
    
#    if map_norm == True:   # inverse norm
#        vol = (1.0-grid)*(gfe_max[0,imap] - gfe_min[0,imap]) + gfe_min[0,imap]
#    else:
#        vol = grid
        
#    nx, ny, nz = vol.shape              #get new dims       
#    vec = grid2vec([nx,ny,nz], vol)     #flatten
    
    #write frag maps to output file
#    out_name = pdb_ids[0]+"."+ map_names_list[imap]
#    write_map(vec, out_path, out_name, center[0,:],
#              res = resolution, n = [nx,ny,nz])
