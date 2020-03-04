import sys
import os
import numpy as np
import torch
from TorchProteinLibrary.FullAtomModel import PDB2CoordsUnordered
from TorchProteinLibrary.FullAtomModel import getBBox

sys.path.append(os.path.join(os.path.dirname(__file__), ".."))
from src.util import pad_mapc, box_face_med, unpad_mapc
from src.mapIO import read_map, write_map



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

    return map_tensor, pad, center



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



def stipOBB(pdb_ids, path, gfe):
    """
    Function to strip out artifacts(high positive GFE values) outside of the bounding box.
    
    """

    nxyz = gfe.shape // 2
    xyz_bb = get_bbox(pdb_ids, path) // 2

    gfe[0:ixL , 0:iyL, 0:izL] = box_med()
    gfe[ixR:nx , iyR:ny, yzR:nz] = box_med()

    return 0


#Below, not used functions!
#def bin_target(target, maxV, minV, scale=1):
#
#    target[target > maxV] = maxV
#    target[target < minV] = minV
#    target = (maxV - target)*scale
#    target = torch.ceil(target).type(torch.cuda.LongTensor)
#
#    return target
#
#
#
#def ubin_target(target, maxV, scale=1):
#
#    target = target / scale
#    target = maxV - target
#
#    return target


if __name__=='__main__':

    

    pdb_ids = ["4f5t"]#"1ycr", "1pw2", "1s4u", "2f6f"]#, "2am9", "3my5_a", "3w8m", "4ic8", "4f5t"]
    path = "/u1/home/tr443/data/fragData/"
    #path = "../../data/"

    #path_list = [path + i + ".pdb" for i in pdb_ids]
    ibbox = get_bbox(pdb_ids, path)
    #print(ibbox)

    dim = int(103)
    map_names_list = ["apolar", "hbacc","hbdon", "meoo", "acec", "mamn"]
    map_path = path + "maps/"
    
    gfe, pad, center = get_target(map_path,
                                  map_names_list,
                                  pdb_ids = pdb_ids,
                                  maxD = dim)

    gap = 1
    dL = dim//2 - ibbox//2 #- gap
    dR = dim//2 + ibbox//2 #+ gap

    i = 0
    for ipdb in pdb_ids:

        gfe1 = gfe.clone()
        gfe1[i, :, 0:dL[i][0],   0:dL[i][1],   0:dL[i][2]]   = 0.0
        gfe1[i, :, dR[i][0]:dim, dR[i][1]:dim, dR[i][2]:dim] = 0.0

        gfe = gfe1# torch.where(gfe < 0.1, gfe, gfe1)
        i = i + 1

                
    gfe = gfe.cpu().detach().numpy()
    for i in range(6):

        grid = gfe[0,i,:,:,:]#.numpy()
        grid = unpad_mapc(grid, pad = pad[0,:])

        nx, ny, nz = grid.shape
        #print(nx,ny,nz)
        new_shape = nx*ny*nz
        vec = np.reshape(grid, new_shape, order="F")

        out_name = pdb_ids[0] + "." + map_names_list[i]+"P"
        write_map(vec, "../output/maps/", out_name, center[0,:], res = 1.0, n = [nx,ny,nz])



