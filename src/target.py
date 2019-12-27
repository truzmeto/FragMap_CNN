import numpy as np
import sys
import os
import torch

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
            _, _, FrE, cent = read_map(maps)                   #ex-f-call
            

            #apply baseline correction
            baseline = box_face_med(FrE)       
            FrE = FrE - baseline
      
            
            #apply centered padding
            FrE, pads = pad_mapc(FrE, maxD, baseline)  #ex-f-call
            
            #convert to tensor
            map_tensor[ibatch,imap,:,:,:] = FrE        #padded_gfe 

            
        pad[ibatch,:] = pads
        center[ibatch,:] = cent
        ibatch += 1

    #convert target maps to torch.cuda
    map_tensor  = torch.from_numpy(map_tensor).float().cuda()
    
    return map_tensor, pad, center 


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
