import os
import sys
import torch
import torch.nn as nn
from src.cnn  import CnnModel
from src.volume import get_volume
from src.mapIO import write_map, greatest_dim
from src.target import get_target
from src.util import grid2vec, unpad_map, sample_batch
import torch.optim as optim
import numpy as np


#model params
lrt = 0.0001
#lrd = 0.0001
wd = 0.00001
max_epoch = 50
batch_size = 2

# physical params
resolution = 1.000
kBT = 0.592 # T=298K, kB = 0.001987 kcal/(mol K)


#pdb_path = "/scratch/tr443/fragmap/data/"                                                          
pdb_path = 'data/'
pdb_ids = ["1ycr", "1pw2", "2f6f",
           "4f5t", "1s4u", "2am9",
           "3my5_a", "3w8m","4ic8"]

#pdb_path_list = [pdb_path+ids+".pdb" for ids in pdb_ids]
map_names_list = ["apolar", "hbacc",
                  "hbdon", "meoo",
                  "acec", "mamn"]

map_path = 'data/maps/' 
#map_path = "/scratch/tr443/fragmap/data/maps/"                                               

dim = greatest_dim(map_path, pdb_ids) + 1

box_size = int(dim*resolution)

batch_list, pdb_list = sample_batch(batch_size,
                                    pdb_ids,
                                    pdb_path,
                                    shuffle = True)

volume = get_volume(path_list = batch_list, 
                    box_size = box_size,
                    resolution = resolution,
                    norm = True,
                    rotate = True)


#get fragmap volumes, padded and baseline corrected
#target, pad, gfe_min, gfe_max = get_target(map_path,
target = get_target(map_path,
                    map_names_list,
                    pdb_ids = pdb_list, #?????????????????????????
                    maxD = dim,
                    cutoff = False,
                    density = False)

# convert to torch.cuda
target = torch.from_numpy(target).float().cuda()

#invoke model
torch.cuda.set_device(0)
model = CnnModel().cuda()
criterion = nn.MSELoss()
#criterion = nn.L1Loss()
optimizer = optim.Adam(model.parameters(), lr = lrt, weight_decay = wd )
#optimizer = optim.SGD(model.parameters(), lr = lrt, momentum = 0.9)


for epoch in range(max_epoch):

    optimizer.zero_grad()
    output = model(volume)
    loss = criterion(output, target)
    loss.backward()
    optimizer.step()
    
    if epoch % 50 == 0:
        print('{0}, {1}'.format(epoch, loss.item()))
            
#save trained parameters        
#out_path = '/scratch/tr443/fragmap/output/'
out_path = 'output/'
torch.save(model.state_dict(), out_path+'net.pth')


#ori = [40.250, -8.472, 20.406] ###### get it from input!!!!!!!!

#for i in range(len(map_names_list)):

#    out_name = pdb_ids[i]+"."+ map_names_list[i]
#    grid = output[0,i,:,:,:].cpu().detach().numpy()
#    grid = unpad_map(grid, xpad = pad[0], ypad = pad[1], zpad = pad[2])

    #convert from Free-E to density 
    #grid[grid <= 0.000] = 0.0001
    #vol = grid #-kBT *np.log(grid)  

    #vol = grid*(gfe_max[i] - gfe_min[i]) + gfe_min[i] 
#    vol = grid
#    nx, ny, nz = grid.shape
# 
#    vec = grid2vec([nx,ny,nz], vol)
#    write_map(vec,
#              out_path,
#              out_name,
#              ori = ori,
#              res = resolution,
#              n = [nx,ny,nz])
