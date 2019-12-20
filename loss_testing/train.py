import os
import sys
import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np

from train import *
from test import *

sys.path.insert(0, '..')
from src.cnn  import CnnModel, CnnModel_Leaky
from src.volume import get_volume
from src.mapIO import greatest_dim, write_map
from src.target import get_target
from src.util import grid2vec, sample_batch, unpad_mapc
from src.loss_fns import *
from visual.scatter import *



def train_model(max_epoch, nsample, batch_size, pdb_ids, pdb_path, 
                box_size, resolution, norm, rand_rotations, 
                map_path, map_names_list, dim, RT, map_norm,
                optimizer, model, criterion,
                out_path, params_file_name):
    '''
    Train cnn model
    output: saved *.pth model (params_file_name)
    '''

    print('Running model', params_file_name[:10])
    
    #perform forward and backward iterations
    for epoch in range(max_epoch):
    
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
                                                            RT = RT,
                                                            density = False,
                                                            map_norm = map_norm)
                    
        #convert target maps to torch.cuda
        target = torch.from_numpy(target).float().cuda()

        if rand_rotations:
            target = grid_rot(target, batch_size, rot_matrix)
            
        
        #perform forward and backward iterations
        #for epoch in range(max_epoch):
            
        optimizer.zero_grad()
        output = model(volume)
        loss = criterion(output, target)
        loss.backward()
        optimizer.step()
            
        if epoch % 40 == 0:
            print('{0},{1},{2}'.format(batches, epoch, loss.item()))
                
    #save trained parameters        
    torch.save(model.state_dict(), out_path+'model/'+params_file_name)
    
    return
