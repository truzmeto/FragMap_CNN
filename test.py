import os
import sys
import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np

from src.convSE import FragMapSE3
from src.volume import get_volume
from src.mapIO import write_map, greatest_dim
from src.target import get_target
from src.util import grid2vec, unpad_mapc, pad_mapc
from src.loss_fns import PenLoss


def get_inp(pdb_ids, resolution, pdb_path, rotate = False):
    """
    This function takes care of all inputs: 
    volume, maps + rotations etc..

    """
    
    norm = True
    map_path = pdb_path + "maps/"                                                                                
    map_names_list = ["apolar", "hbacc","hbdon", "meoo", "acec", "mamn"]
    dim = greatest_dim(map_path, pdb_ids) + 1
    box_size = int(dim*resolution)
    batch_list = [pdb_path + ids + ".pdb" for ids in pdb_ids]

    with torch.no_grad():

        volume, _ = get_volume(path_list = batch_list, 
                               box_size = box_size,
                               resolution = resolution,
                               norm = norm,
                               rot = False,
                               trans = False)
        
        target, pad, center = get_target(pdb_path,
                                map_names_list,
                                pdb_ids = pdb_ids,
                                maxD = dim)

        
        if rotate:
            irot = torch.randint(0, 24, (1,)).item()
            volume = Rot90Seq(volume, iRot = irot)
            target = Rot90Seq(target, iRot = irot)
                    
    return volume, target



def output_maps(output, pad, resolution, ipdb, out_path, map_name_list):

    for imap in range(len(map_names_list)):
        
        grid = output[0,imap,:,:,:].cpu().detach().numpy()
        grid = unpad_mapc(grid, pad = pad[0,:])
        
        nx, ny, nz = grid.shape              #get new dims
        vec = grid2vec([nx,ny,nz], grid)     #flatten
        
        out_name = ipdb + "."+ map_names_list[imap] + "P"
        write_map(vec, out_path + "maps/", out_name, center[0,:],
                  res = resolution, n = [nx,ny,nz])
        
    return None

if __name__=='__main__':

    
    resolution = 1.000
    RT = 0.59248368    # T=298.15K, R = 0.001987204 kcal / (mol K)
    istate_load = 4000
    
    pdb_path = '../../data/'
    pdb_ids = ["1ycr", "1pw2", "2f6f","4ic8", "1s4u", "2am9", "3my5_a", "3w8m","4f5t"]
    map_names_list = ["apolar", "hbacc","hbdon", "meoo", "acec", "mamn"]
    map_path = '../../data/maps/'
    out_path = 'output/'
    map_names_listM = [m + "M" for m in map_names_list]

    
    params_file_name = str(istate_load)+'net_params'
    torch.cuda.set_device(0)
    model = FragMapSE3().cuda()
    model.load_state_dict(torch.load(out_path +params_file_name))#+".pth"))
    model.eval() #Needed to set into inference mode
    criterion = PenLoss()

    
    for ipdb in pdb_ids:
        
        #get inp data
        volume, test_map, pad, center =  get_inp(pdb_ids, resolution, pdb_path, rotate = False)
        
        #invoke model
        output = model(volume)
        loss = criterion(output, test_map, thresh = 2.0)
        
        ######################################################
        print("Testing Loss", loss.item(), ipdb)
     
        #output predicted maps and modified input maps
        output_maps(output, pad, resolution, ipdb, out_path, map_name_list)
        output_maps(test_map, pad, resolution, ipdb, out_path, map_name_listM)


