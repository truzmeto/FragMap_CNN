import os
import sys
import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import gc

#import functions from src
from src.cnn  import CnnModel
from src.volume import get_volume, grid_rot
from src.mapIO import greatest_dim, write_map
from src.target import get_target
from src.util import grid2vec, sample_batch, unpad_mapc


#model params
lrt = 0.0001
#lrd = 0.0001
wd = 0.00001
max_epoch = 20
batch_size = 4 #number of structures in a batch

norm = True
map_norm = False
nsample = 1


#physical params
resolution = 1.000
RT = 0.59248368 # T=298.15K, R = 0.001987204 kcal/(mol K)

pdb_path = 'data/'
#pdb_path = "/scratch/tr443/fragmap/data/"                                                          
pdb_ids = ["1ycr", "1pw2", "2f6f","4ic8", "1s4u", "2am9", "3my5_a", "3w8m"]#,"4f5t"]

map_names_list = ["apolar", "hbacc","hbdon", "meoo", "acec", "mamn"]
map_path = 'data/maps/' 
#map_path = "/scratch/tr443/fragmap/data/maps/"                                               

#out_path = '/scratch/tr443/fragmap/output/'
out_path = 'output/'

dim = greatest_dim(map_path, pdb_ids) + 1
box_size = int(dim*resolution)

print("#Box Dim = ",box_size)
params_file_name = 'net_params.pth'

#invoke model
torch.cuda.set_device(0)
model = CnnModel().cuda()
#criterion = nn.MSELoss()
criterion = nn.SmoothL1Loss() #nn.L1Loss()
optimizer = optim.Adam(model.parameters(), lr = lrt, weight_decay = wd )
#optimizer = optim.SGD(model.parameters(), lr = lrt, momentum = 0.9)
rand_rotations = True


print("#batch_id", "epoch", "Loss", "pdb_list")

for epoch in range(1, max_epoch+1):
    for batches in range(nsample):
        batch_list, pdb_list = sample_batch(batch_size,
                                            pdb_ids,
                                            pdb_path,
                                            shuffle = True)

        with torch.no_grad():
            #get batch volume tensor
            volume, rot_matrix = get_volume(path_list = batch_list, 
                                            box_size = box_size,
                                            resolution = resolution,
                                            norm = norm,
                                            rot = rand_rotations,
                                            trans = False)
        
            #get target map tensor torch.cuda()
            target, _, _ = get_target(map_path,
                                      map_names_list,
                                      pdb_ids = pdb_list,
                                      maxD = dim)

            
            
            #target maps preprocessing here!
            
            
            if rand_rotations:
                target = grid_rot(target, batch_size, rot_matrix)

        #############################################################
        
        
        optimizer.zero_grad()
        output = model(volume)
        loss = criterion(output, target)

        loss.backward()
        optimizer.step()

        #print("gpu_mem: allocated",                                                                     
        #      str(torch.cuda.memory_allocated(device=None)/1000000)+"Mbs" )                             

       
        #print("gpu mem. empty",                                                         
        #      str((torch.cuda.memory_cached()-torch.cuda.memory_allocated())/1000000)+"Mbs" )    

                
        if epoch % 10 == 0:
            print('{0}, {1}, {2}, {3}'.format(batches, epoch, loss.item(), pdb_list))

            
        #gc.collect()
        #del volume, target, output
        #torch.cuda.empty_cache() #
        #print("gpu_cacched_mem",                                                         
        #      str(torch.cuda.memory_cached(device=None)/1000000)+"Mbs" )                            

    
    if epoch % 50 == 0:
        torch.save(model.state_dict(), out_path + str(epoch) + params_file_name)

    
