import numpy as np
import sys
import os
import torch
from TorchProteinLibrary.FullAtomModel import PDB2CoordsUnordered
from TorchProteinLibrary.FullAtomModel import getBBox

sys.path.append(os.path.join(os.path.dirname(__file__), ".."))
from src.util import pad_mapc,  box_face_med
from src.mapIO import read_map


def get_target(map_path, map_names, pdb_ids, maxD):
    """
    This function invokes necessary frag maps, pads them
    and returns them with required tensor dimension.
    

    """

    batch_size = len(pdb_ids)
    n_maps = len(map_names)
    map_tensor = np.zeros(shape = (batch_size, n_maps, maxD, maxD, maxD))

    pad = np.empty(shape = [batch_size, 3], dtype=int)
    center = np.empty(shape = [batch_size, 3], dtype=float)

    ibatch = 0
    for batch in pdb_ids:
        for imap in range(n_maps):

            maps = map_path + batch +"."+map_names[imap] + ".gfe.map"
            _, _, FrE, cent = read_map(maps)                   
            

            #apply baseline correction
            baseline = box_face_med(FrE)       
            FrE = FrE - baseline
                  
            #apply centered padding
            FrE, pads = pad_mapc(FrE, maxD, 0.00)  #ex-f-call
            
            #convert to tensor
            map_tensor[ibatch,imap,:,:,:] = FrE        #padded_gfe 

            
        pad[ibatch,:] = pads
        center[ibatch,:] = cent
        ibatch += 1

        
   
    #convert target maps to torch.cuda
    map_tensor  = torch.from_numpy(map_tensor).float().cuda()
    
    return map_tensor, pad, center 


def get_bbox(pdb_ids, path):
    """
    Function returns bbox size for all
    input pdbs. 
    """
    
    path_list = [path + i + ".pdb" for i in pdb_ids]
    pdb2coords = PDB2CoordsUnordered()
    coords, _, resnames, _, atomnames, num_atoms = pdb2coords(path_list)
    a, b = getBBox(coords, num_atoms)
    c = a - b

    return torch.abs(c).int().numpy()


def stipOBB(pdb_ids, path, gfe):
    """

    to be continued!!!!!!!!!!!!!
    """
    nxyz = np.array(gfe.shape) // 2 
    xyz_bb = get_bbox(pdb_ids, path) // 2
    
    #ix = nx // 2 -  xyz[:,0] + 2
    #iy = ny // 2 -  xyz[:,1] + 2
    #iz = nz // 2 -  xyz[:,2] + 2
    ixyz = xyz_bb - nxyz + 2

    gfe[0:ixL , 0:iyL, 0:izL] = box_med()
    gfe[ixR:nx , iyR:ny, yzR:nz] = box_med()
    
    
    
    return 0



#Below, not used functions!
def bin_target(target, maxV, minV, scale=1):
    
    target[target > maxV] = maxV
    target[target < minV] = minV
    target = (maxV - target)*scale
    target = torch.ceil(target).type(torch.cuda.LongTensor)

    return target



def ubin_target(target, maxV, scale=1):

    target = target / scale
    target = maxV - target
  
    return target




def get_target1(map_path, map_names, pdb_ids, maxD, RT,
               density = False, map_norm = False):
    """
    This function invokes necessary frag maps, pads them
    and returns them with required tensor dimension.
    
    """

    batch_size = len(pdb_ids)
    n_maps = len(map_names)
    map_tensor = np.zeros(shape = (batch_size, n_maps, maxD, maxD, maxD))

    gfe_min = np.empty(shape = [batch_size, n_maps], dtype = float)
    gfe_max = np.empty(shape = [batch_size, n_maps], dtype = float)
    pad = np.empty(shape = [batch_size, 3], dtype = int)
    center = np.empty(shape = [batch_size, 3], dtype = float)

    baseline = np.empty(shape = [batch_size, n_maps], dtype = float)
    ibatch = 0


    for batch in pdb_ids:
        for imap in range(n_maps):

            maps = map_path + batch +"."+map_names[imap] + ".gfe.map"
            _, _, FrE, cent = read_map(maps)                   #ex-f-call
            
            
            # apply baseline correction
            baseline[ibatch, imap] = box_face_med(FrE)       #ex-f-call
            FrE = FrE - baseline[ibatch, imap]
      
            #apply cutoff to Frag Free Energy
            #if cutoff == True:
            #    FrE[FrE > 0] = 0.0 
                
            if density == True:                                #convert to density 
                FrE = np.exp(-FrE / RT) 
            else:                                              #or return GFE maps
                
                if map_norm == True: #min-max normalize maps
                    gfe_min[ibatch, imap] = FrE.min() 
                    gfe_max[ibatch, imap] = FrE.max()
                    FrE  =  (FrE - gfe_min[ibatch,imap]) / (gfe_max[ibatch,imap] - gfe_min[ibatch,imap])
             
                    
            #apply centered padding
            FrE, pads = pad_mapc(FrE, maxD, baseline[ibatch,imap])   #ex-f-call
            
            #convert to tensor
            map_tensor[ibatch,imap,:,:,:] = FrE #pad_dens 
            
        
        pad[ibatch,:] = pads
        center[ibatch,:] = cent
        ibatch += 1

    return map_tensor, pad, gfe_min, gfe_max, center 
