import sys
import os
import numpy as np
import torch
from TorchProteinLibrary.FullAtomModel import PDB2CoordsUnordered
from TorchProteinLibrary.FullAtomModel import getBBox

sys.path.append(os.path.join(os.path.dirname(__file__), ".."))
from src.util import pad_mapc, box_face_med, unpad_mapc
from src.mapIO import read_map, write_map



def get_target(path, map_names, pdb_ids, maxD):
    """
    This function invokes necessary frag maps, pads them
    and returns them with required tensor dimension.


    """

    map_path = path + "maps/"
    batch_size = len(pdb_ids)
    n_maps = len(map_names)
    map_tensor = np.zeros(shape = (batch_size, n_maps, maxD, maxD, maxD))

    pad = np.empty(shape = [batch_size, 3], dtype=int)
    center = np.empty(shape = [batch_size, 3], dtype=float)

    ibatch = 0
    for ipdb in pdb_ids:
        for imap in range(n_maps):

            maps = map_path + ipdb + "." + map_names[imap] + ".gfe.map"
            _, _, FrE, cent = read_map(maps)

            #baseline = box_face_med(FrE)
            baseline = np.median(FrE)
            FrE = FrE - baseline

            #apply centered padding
            FrE, pads = pad_mapc(FrE, maxD, 0.00)  #ex-f-call

            #convert to tensor
            map_tensor[ibatch,imap,:,:,:] = FrE        #padded_gfe

        pad[ibatch,:] = pads
        center[ibatch,:] = cent
        ibatch += 1

    #convert target maps to torch.cuda
    map_tensor = torch.from_numpy(map_tensor).float().cuda()

    #strip out high gfe values that reside outside of bbox
    gfe = stipOBB(pdb_ids, path, map_tensor, gfe_thresh = -0.2, gap = 0)

    return gfe, pad, center



def get_bbox(pdb_ids, path):
    """
    Function returns bbox size for all
    input pdbs.
    ---------------------------

    Input:   pdb_ids - list of pdb names
             path    - path to pdb files directory 

    Output:  
    
    """

    path_list = [path + i + ".pdb" for i in pdb_ids]
    pdb2coords = PDB2CoordsUnordered()
    coords, _, resnames, _, atomnames, num_atoms = pdb2coords(path_list)
    r_min, r_max = getBBox(coords, num_atoms)
    bbox_dims = torch.abs(r_min - r_max).int()
    
    return bbox_dims


def stipOBB(pdb_ids, path, gfe, gfe_thresh = 0.1, gap = 1):
    """
    Function to strip out artifacts(high positive GFE values)
    outside of the bounding box. It performs above cleanng
    satisfying two necessary conditions:
    1. GFE values must reside outside of bounding box
    2. GFE values must be greater than the threshold(0.1)
  
    """

    ibbox = get_bbox(pdb_ids, path)
    dims = gfe.shape
        
    if (dims[2] != dims[3]) or (dims[2] != dims[3]) or (dims[3] != dims[4]):
        raise ValueError("Box dims is not cubic!")
 
    dim = dims[-1]
    dL = dim//2 - ibbox//2 - gap
    dR = dim//2 + ibbox//2 + gap
    
    i = 0
    for ipdb in pdb_ids:
        
        gfe1 = gfe.clone()
        gfe1[i, :, 0:dL[i][0],  :, :] = 0.0
        gfe1[i, :, :, 0:dL[i][1], :]  = 0.0
        gfe1[i, :, :, :, 0:dL[i][2]]  = 0.0
        
        gfe1[i,  :, dR[i][0]:dim, :, :] = 0.0
        gfe1[i, :, :, dR[i][1]:dim, :]  = 0.0
        gfe1[i, :, :, :, dR[i][2]:dim]  = 0.0
           
        gfe = torch.where(gfe < gfe_thresh, gfe, gfe1) # keep gfe values less than 'gfe_thresh'
                                                       # and replace the rest with gfe1 values
                                                       # that acts upon valuse outside of bbox
        i = i + 1
        
    return gfe


if __name__=='__main__':
    print("I feel good!")
    
