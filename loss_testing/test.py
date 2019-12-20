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


def test_model(pdb_path, pdb_ids, map_names_list, map_path, norm,
               batch_list, box_size, resolution,
               test_indx, dim, RT, map_norm,
               out_path, params_file_name, criterion, pmap_dir):
    
    volume , _ = get_volume(path_list = batch_list, 
                        box_size = box_size,
                        resolution = resolution,
                        norm = norm,
                        rot = False,
                        trans = False)

    #get testing map tensor
    #map_norm = False
    test_map, pad, gfe_min, gfe_max, center = get_target(map_path,
                                                map_names_list,
                                                pdb_ids = [pdb_ids[test_indx]],
                                                maxD = dim,
                                                RT = RT,
                                                density = False,
                                                map_norm = map_norm
                                                )

    #convert target maps to torch.cuda
    test_map = torch.from_numpy(test_map).float().cuda()


    #-------------------------------------------------------------
    #invoke model
    torch.cuda.set_device(0)
    #model = load_model(out_path, params_file_name)

    model = CnnModel_Leaky().cuda()

    model.load_state_dict(torch.load(out_path+"model/"+params_file_name))

    model.eval() #Needed to set into inference mode
    output = model(volume)


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
        out_name = pdb_ids[test_indx]+"."+ map_names_list[imap]+"_o"
        write_map(vec, pmap_dir, out_name, center[0,:],
                res = resolution, n = [nx,ny,nz])
