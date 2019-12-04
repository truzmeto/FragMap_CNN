import numpy as np
import sys
import os

sys.path.append(os.path.join(os.path.dirname(__file__), ".."))
from src.util import pad_map
from mapIO import read_map


def get_target(map_path, map_names,
               pdb_ids, batch_size,
               dim, cutoff = False,
               density = False):
    """
    This function invokes necessary frag maps, pads them
    and returns them with required tensor dimension.
    
    """
    
    map_path_list = []
    for iname in map_names:
        if iname == "excl":
            map_tail = ".map"
        else:
            map_tail = ".gfe.map"
        map_path_list.append(map_path + pdb_ids + "." + iname + map_tail)#????????????????/

            
    n_maps = len(map_names)
    map_tensor = np.zeros(shape = (batch_size, n_maps, dim, dim, dim))
    kBT = 0.592

    gfe_min = []
    gfe_max = []


    for ibatch in range(batch_size):
        for imap in range(n_maps):

            _, _, FrE = read_map(map_path_list[imap])      #in-f-call
            
            #apply cutoff to Frag Free Energy
            if cutoff == True:
                FrE[FrE > 0] = 0.0 
                    
            if density == True: #convert to density 
                dens = np.exp(-FrE / kBT) 
            else:               #normalize GFE maps
                gfe_min.append(FrE.min())
                gfe_max.append(FrE.max())
                dens = (FrE - gfe_min[imap]) / (gfe_max[imap] - gfe_min[imap])

            #apply padding
            pad_dens, xpad, ypad, zpad = pad_map(dens)   #ex-f-call
            
            #convert to tensor
            map_tensor[ibatch, imap,:,:,:] = pad_dens
       
            pad = [xpad, ypad, zpad]

    return map_tensor, pad, gfe_min, gfe_max  
