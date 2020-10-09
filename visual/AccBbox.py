import os
import sys
import torch
import numpy as np

from TorchProteinLibrary.FullAtomModel import PDB2CoordsUnordered
from TorchProteinLibrary.FullAtomModel import getBBox


def box_face_error(pred, target):
    """
    This function calculates GFE error between predicted
    and original fragmaps over voxels along the 6 faces of the
    grid of given dim (nx, ny, nz).
    
    """

    nx, ny, nz = pred.shape
    mx, my, mz = target.shape

    if ( mx != nx) or (ny != my) or (nz != mz):
        raise ValueError("Predicted and original grid dimentions are different!")

    
    dface_x = (pred[nx-1,:,:] - target[nx-1,:,:]).pow(2).sum() + (pred[0,:,:] - target[0,:,:]).pow(2).sum()
    dface_y = (pred[1:nx-1,ny-1,:] - target[1:nx-1,ny-1,:]).pow(2).sum() + (pred[1:nx-1,0,:] - target[1:nx-1,0,:]).pow(2).sum()
    dface_z = (pred[1:nx-1,1:ny-1,nz-1] - target[1:nx-1,1:ny-1,nz-1]).pow(2).sum() + (pred[1:nx-1,1:ny-1,0] - target[1:nx-1,1:ny-1,0]).pow(2).sum()

    
    #number of voxels on face, simple algebra
    n_face_vox = nx*ny*nz - (nx-2)*(ny-2)*(nz-2)
    face_err = dface_x + dface_y + dface_z
    face_err = face_err.sqrt() / n_face_vox

    return face_err

def DistErr(pred, target):
    """
    Construct bounding box face error VS distance
    dist = (a_bb + b_bb + c_bb) / 6

    """

    dim =  pred.shape
    nx, ny, nz = dim #+ 2
    n = np.min(dim)//2   

    err = []
    dist = []
    
    for i in range(n):

        pred_slice = pred[i:nx-i, i:ny-i, i:nz-i]
        target_slice = target[i:nx-i, i:ny-i, i:nz-i]
                
        err.append(box_face_error(pred_slice, target_slice))
        dist.append(np.mean(pred_slice.shape) // 2) 

    return dist, err
    
if __name__=='__main__':
    
    import matplotlib.pyplot as plt
    sys.path.append(os.path.join(os.path.dirname(__file__), ".."))
    from src.mapIO import read_map 
    
    pdb_ids = ["1ycr","1pw2","2f6f", "4f5t", "2am9", "3my5_a", "3w8m", "4ic8"] 
    path = "../output/maps/"#"/u1/home/tr443/data/fragData/maps/"
    frag_names = ["apolar", "hbacc","hbdon", "meoo", "acec", "mamn"]
    
    pdb_id = pdb_ids[1]
    path_list = [path + pdb_id + "." + i + ".gfe.map" for i in frag_names]
    path_listP = [path + pdb_id + "." + i + "P.gfe.map" for i in frag_names]

    for i, imap in enumerate(path_list):
        
        _, n_cells, target, _ = read_map(imap)
        _, _, pred, _ = read_map(path_listP[i])
                
        pred = torch.from_numpy(pred).float()
        target = torch.from_numpy(target).float()
        dist, err = DistErr(pred, target)

        plt.plot(dist, err, lw=2, ls='--')
        
    box_size = np.mean(n_cells) // 2
    plt.title(pdb_id + " ," + str(int(box_size)))
    plt.legend(frag_names)
    plt.show()
    
