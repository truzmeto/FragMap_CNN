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
from src.rot24 import Rot90Seq


#model params
lrt = 0.0001
#lrd = 0.0001
wd = 0.00001
max_epoch = 20#00
batch_size = 1 #number of structures in a batch
norm = True


resolution = 1.000
RT = 0.59248368 # T=298.15K, R = 0.001987204 kcal/(mol K)

pdb_path = 'data/'
#pdb_path = "/scratch/tr443/fragmap/data/"                                                          
pdb_ids = ["1ycr"]#, "1pw2", "2f6f","4ic8", "1s4u", "2am9", "3my5_a", "3w8m"]#,"4f5t"]
#map_names_list = ["apolar" "hbacc", "hbdon", "meoo", "acec", "mamn"]
map_names_list = ["hbacc"]
map_path = 'data/maps/' 
out_path = 'output/'

dim = greatest_dim(map_path, pdb_ids) + 1
box_size = int(dim*resolution)

print("#Box Dim = ",box_size)
params_file_name = 'net_params.pth'

#invoke model
torch.cuda.set_device(0)
model = CnnModel().cuda()
#model.load_state_dict(torch.load(out_path +"params/apolar/"+"900"+params_file_name))
criterion = nn.CrossEntropyLoss() 
optimizer = optim.Adam(model.parameters(), lr = lrt, weight_decay = wd )

print("#batch_id", "epoch", "Loss", "pdb_list")

nsample = len(pdb_ids) // batch_size
nrots = 1
istart = 1
iend = max_epoch + 1
for epoch in range(istart, iend):
    
    #sequentially samples batches
    for batches in range(nsample):
        j = batches * batch_size
        pdb_list = pdb_ids[j: j + batch_size]
        batch_list = [pdb_path+ids+".pdb" for ids in pdb_list]

        #rotate each batch 24 times
        for irot in range(nrots):
            #with torch.no_grad():
            #get batch volume tensor
            volume, _ = get_volume(path_list = batch_list, 
                                   box_size = box_size,
                                   resolution = resolution,
                                   norm = norm,
                                   rot = False,
                                   trans = False)
            
            #get target map tensor torch.cuda()
            target, _, _ = get_target(map_path,
                                      map_names_list,
                                      pdb_ids = pdb_list,
                                      maxD = dim)
            
            
            ##########--- Target maps preprocessing here! ---############
            ## had to pull it here to avoid messing with src :) ####
            
            #offset, and discretize
           # offs = 3.0
           # target[target > offs] = offs
           # target[target < -2.0] = -2.0
            #target = offs - target
            target = torch.mul(target,2)
            target = torch.floor(target).type(torch.cuda.LongTensor)

            #apply 24 rotations, works with torch.cuda()
            volume = Rot90Seq(volume, iRot = irot)
            target = Rot90Seq(target, iRot = irot)
            
            
            optimizer.zero_grad()
            output = model(volume)
            
            target = target.squeeze(1)
            loss = criterion(output, target)
            loss.backward()
            optimizer.step()
            
            
            #if epoch % 5 == 0:
            print('irot:{0},ibatch:{1},epoch:{2},loss:{3},pdb_list:{4}'.format(irot,
                                                                    batches, epoch, loss.item(), pdb_list))

            
            #gc.collect()
            #del volume, target, output
            #torch.cuda.empty_cache() #
            #print("gpu_cacched_mem",                                                         
            #      str(torch.cuda.memory_cached(device=None)/1000000)+"Mbs" )                            
    
    if epoch % 50 == 0:
        torch.save(model.state_dict(), out_path + str(epoch) + params_file_name)
        
        
