import os
import sys
import torch
import torch.nn as nn
from src.cnn  import CnnModel_Leaky
from src.volume import get_volume
from src.mapIO import greatest_dim, write_map
from src.target import get_target
from src.util import grid2vec, sample_batch, unpad_mapc, save_model
from src.augment import get_24, get_random_rotation
import torch.optim as optim
import numpy as np
import pickle as pkl


#model params
lrt = 0.0001
#lrd = 0.0001
wd = 0.00001
max_epoch = 1000
batch_size = 1 #number of structures in a batch
norm = True
map_norm = False
nsample = 10
i= 0
chkpt_step = 500

#physical params
resolution = 1.000
kBT = 0.592 # T=298K, kB = 0.001987 kcal/(mol K)

pdb_path = 'data/'
pdb_ids = ["1ycr", "1pw2", "2f6f", "4f5t", "1s4u", "2am9", "3my5_a", "3w8m"]#,"4ic8"]
map_names_list = ["apolar", "hbacc","hbdon", "meoo", "acec", "mamn"]
map_path = 'data/maps/' 
out_path = 'output/'
#pdb_path = "/scratch/tr443/fragmap/data/"                                                          
#map_path = "/scratch/tr443/fragmap/data/maps/"                                               
#out_path = '/scratch/tr443/fragmap/output/'

dim = greatest_dim(map_path, pdb_ids) + 1
box_size = int(dim*resolution)
params_file_name = 'net_params_1k_xnorm_hotspot'

#invoke model
torch.cuda.set_device(0)
model = CnnModel_Leaky().cuda()
criterion = nn.MSELoss()
#criterion = nn.L1Loss()
optimizer = optim.Adam(model.parameters(), lr = lrt, weight_decay = wd )
#optimizer = optim.SGD(model.parameters(), lr = lrt, momentum = 0.9)


for epoch in range(max_epoch):
    # # Get rotated volume and target 
    # rot_volume, rot_gfe= get_random_rotation(volume, target)
    #perform forward and backward iterations
    for batches in range(nsample):
        #sample batch list from all structures
        batch_list, pdb_list = sample_batch(batch_size,
                                            pdb_ids,
                                            pdb_path,
                                            shuffle = True)
        #get batch volume tensor
        volume, _ = get_volume(path_list = batch_list, 
                            box_size = box_size,
                            resolution = resolution,
                            norm = norm,
                            rot = False)
        
        #get target map tensor
        target, pad, gfe_min, gfe_max, ori = get_target(map_path,
                                                    map_names_list,
                                                    pdb_ids = pdb_list,
                                                    maxD = dim,
                                                    kBT = kBT,
                                                    density = False,
                                                    map_norm = map_norm)
        
        # #convert target maps to torch.cuda

        # rot_vol, rot_target = get_random_rotation(volume, torch.tensor(target))

        rot_target = torch.from_numpy(target).float().cuda()

        optimizer.zero_grad()
        output = model(volume)
        loss = criterion(output, rot_target)
        loss.backward()
        optimizer.step()
      
        if batches % 20 == 0:
            print('{0},{1},{2}'.format(batches, epoch, loss.item()))
    if epoch == i*chkpt_step:
        save_model(model, out_path, params_file_name)
        i+=1

# save trained parameters        
save_model(model, out_path, params_file_name)
