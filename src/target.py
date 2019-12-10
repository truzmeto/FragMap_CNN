import numpy as np
import sys
import os

sys.path.append(os.path.join(os.path.dirname(__file__), ".."))
from src.util import pad_map, box_face_ave
from src.mapIO import read_map, greatest_dim


def get_target(map_path, map_names, pdb_ids, maxD, kBT,
               cutoff = False, density = False,
               map_norm = False):
    """
    This function invokes necessary frag maps, pads them
    and returns them with required tensor dimension.
    
    """

    batch_size = len(pdb_ids)
    n_maps = len(map_names)
    map_tensor = np.zeros(shape = (batch_size, n_maps, maxD, maxD, maxD))

    gfe_min = np.empty(shape = [batch_size, n_maps])
    gfe_max = np.empty(shape = [batch_size, n_maps])

    baseline = np.empty(shape = [batch_size, n_maps])
    pad = []
    ibatch = 0

    #for ibatch in range(batch_size):
    for batch in pdb_ids:
        for imap in range(n_maps):

            maps = map_path + batch +"."+map_names[imap] + ".gfe.map"
            _, _, FrE = read_map(maps)      #ex-f-call


            #apply baseline correction
            #baseline[ibatch, imap] = box_face_ave(FrE)         #ex-f-call
            #FrE = FrE - baseline[ibatch, imap]
      
            #apply cutoff to Frag Free Energy
            if cutoff == True:
                FrE[FrE > 0] = 0.0 
                
            if density == True:                     #convert to density 
                dens = np.exp(-FrE / kBT) 
            else:                           #return GFE maps

                if map_norm == True: #min-max normalize maps
                    gfe_min[ibatch, imap] = FrE.min()
                    gfe_max[ibatch, imap] = FrE.max()
                    dens = (FrE - gfe_min[ibatch,imap]) / (gfe_max[ibatch,imap] - gfe_min[ibatch,imap])
                else:
                    dens = FrE
                
            #apply padding
            pad_dens, xpad, ypad, zpad = pad_map(dens, maxD)   #ex-f-call
            
            #convert to tensor
            map_tensor[ibatch,imap,:,:,:] = pad_dens 
        
        ibatch+=1
        pad.append([xpad, ypad, zpad]) 

    return map_tensor, pad, gfe_min, gfe_max #, baseline  
