import numpy as np
import sys
import os

sys.path.append(os.path.join(os.path.dirname(__file__), ".."))
from src.util import pad_map, box_face_ave
from mapIO import read_map


def get_target(map_path, map_names,
               pdb_ids, batch_size,
               dim, cutoff = False,
               density = False):
    """
    This function invokes necessary frag maps, pads them
    and returns them with required tensor dimension.
    
    """
    
    n_maps = len(map_names)
    map_tensor = np.zeros(shape = (batch_size, n_maps, dim, dim, dim))
    kBT = 0.592

    gfe_min = []#!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!
    gfe_max = []


    for ibatch in range(batch_size):
        for imap in range(n_maps):

            maps = map_path + pdb_ids[ibatch] + map_names[imap] + map_tail
            _, _, FrE = read_map(maps)      #ex-f-call

            #apply baseline correction
            FrE = box_face_ave(FrE)
            
            #apply cutoff to Frag Free Energy
            if cutoff:
                FrE[FrE > 0] = 0.0 
                    
            if density: #convert to density 
                dens = np.exp(-FrE / kBT) 
            else:               #normalize GFE maps
                gfe_min.append(FrE.min())
                gfe_max.append(FrE.max())
                dens = (FrE - gfe_min[imap]) / (gfe_max[imap] - gfe_min[imap])
                #dens = 1.0 - dens

            #apply padding
            pad_dens, xpad, ypad, zpad = pad_map(dens)   #ex-f-call
            
            #convert to tensor
            map_tensor[ibatch, imap,:,:,:] = pad_dens
       
            pad = [xpad, ypad, zpad]

    return map_tensor, pad, gfe_min, gfe_max  
