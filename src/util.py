import numpy as np
import random 
import sys  




def sample_batch(batch_size, pdb_ids, pdb_path, shuffle = True):
    """
    Function to produce random sample of pdb ids from a given list.
    This ids are joined with the path to actual file location
    and used to access the data. 

    Input:  batch_size  - number of input files in a batch, dtype=list
            pdb_ids     - full list of pdbs to sample from, dtype=list
            pdb_path    - path to file location, dtype=string 
          
    Output: batch_list  - list of selected files as full path, dtype=list
            pdb_list    - list of selected pdbs, dtype=list

    """      

    if batch_size > len(pdb_ids):
        raise ValueError("Batch size must be less or equal to #pdbs")
        
    if shuffle:
        random.shuffle(pdb_ids)

    pdb_list = random.sample(pdb_ids, batch_size)
    batch_list = [pdb_path+ids+".pdb" for ids in pdb_list]
    
    return batch_list, pdb_list



def vec2grid(n, vec):
    """
    This function transforms 1D vec. into
    tensor (nx,ny,nz) data structure
    """
    nx, ny, nz = n 
    grid = np.reshape(vec, newshape = (nx,ny,nz), order = "F")
    
    return grid


def grid2vec(n, grid):
    """
    This function transforms 3D grid.(nx,ny,nz) into
    vector (nx*ny*nz) 
    """
    nx, ny, nz = n 
    vec = np.reshape(grid, newshape = nx*ny*nz, order = "F")

    return vec



def pad_mapc(dens, maxD, pad_val):
    """
    This function pads np.ndarray of size(nx, ny, nz)
    into a cubic box according to maxD provided.
    Zero padding applied at both sides(RHS and LHS) to
    produce padded volume, np.ndarray of size(maxD,maxD,maxD).
    
    """

    pad = maxD - np.array(dens.shape, dtype = int) 
    if any(pad < 0):
        raise ValueError("Pad length can't be negative", pad) 
        #print("Applying unpadding!")
        #return  unpad_mapc(dens, np.abs(pad)), np.array([0,0,0],dtype=int)
        
    else:
        
        pl = pad//2
        pr = pad - pl
        
        #Pad both sides, if pad length is odd, then pad_l-even, and pad_r-odd  
        dens = np.pad(dens,
                      pad_width = ((pl[0],pr[0]), (pl[1],pr[1]), (pl[2],pr[2])),
                      mode = 'constant', constant_values = pad_val) 
        
        return dens, pad



def unpad_mapc(dens, pad):
    """
    This function unpads the volume by
    slicing out the original part from
    padded volume. It inverts the operation
    done by 'pad_mapc' function.
    
    Input:  tensor -- np.ndarray(shape=(nx,ny,nz))
            pad    -- np.array(px,py,pz)

    Output: tensor -- np.ndarray(shape=(nx-px,ny-py,nz-pz))
    """
    nx, ny, nz = dens.shape
  
    pl = pad // 2
    pr = pad - pl
    pdens = dens[pl[0]:nx-pr[0],
                 pl[1]:ny-pr[1],
                 pl[2]:nz-pr[2]]
    return pdens


def box_face_ave(grid):
    """
    This function calculates GFE average over
    voxels along the 6 faces of the grid. n.ndarray
    slicing and some math is used to have fast calc. 
    
    """
    
    nx, ny, nz = grid.shape
    
    face_x = grid[nx-1,:,:].sum() + grid[0,:,:].sum() 
    face_y = grid[1:nx-1,ny-1,:].sum() + grid[1:nx-1,0,:].sum() 
    face_z = grid[1:nx-1,1:ny-1,nz-1].sum() + grid[1:nx-1,1:ny-1,0].sum() 

    #number of voxels on face, simple algebra 
    n_face_vox = nx*ny*nz - (nx-2)*(ny-2)*(nz-2)
    face_ave = (face_x + face_y + face_z) / n_face_vox

    return face_ave



def box_face_med(grid):
    """
    This function calculates GFE median over
    voxels along 6 faces of the grid.

    Benchmark: dim = (300, 300, 300) grid only takes 15 milli secs.
    """

    nx, ny, nz = grid.shape
        
    #exact slice
    face = np.concatenate((grid[nx-1,:,:], grid[0,:,:],
                           grid[1:nx-1,ny-1,:], grid[1:nx-1,0,:], 
                           grid[1:nx-1,1:ny-1,nz-1], grid[1:nx-1,1:ny-1,0]), axis = None)  
       
    return np.median(face)



if __name__=='__main__':
    print("I need coffe! ")
