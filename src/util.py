import numpy as np
import random 
import sys  
import torch
import pickle


RT =  0.00198720425864083 * 298.15

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



def pad_mapc(dens, maxD):
    """
    This function pads np.ndarray of size(nx,ny,nz)
    into a cubic box according to maxD provided.
    Zero padding applied at both sides(RHS and LHS) to
    produce padded volume, np.ndarray of size(maxD,maxD,maxD).
    
    """

    pad = maxD - np.array(dens.shape,dtype=int) 
    if any(pad < 0):
        raise ValueError("Pad length can't be negative", pad) 
    
    pl = pad//2
    pr = pad - pl
    
    #Pad both sides, if pad length is odd, then pad_l-even, and pad_r-odd  
    pdens = np.pad(dens,
                   pad_width = ((pl[0],pr[0]), (pl[1],pr[1]), (pl[2],pr[2])),
                   mode = 'constant') #zero padding by default

    return pdens, pad


def unpad_mapc(dens, pad):
    """
    This function unpads the volume by
    slicing out the original part from
    padded volume. It inverts the operation
    of done by 'pad_mapc' function.
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

    Benchmark: dim = (300,300,300) grid only takes 15 milli secs.
    """    
    nx, ny, nz = grid.shape
    
    #median calc. don't care about repeated voxels
    #face = np.concatenate((grid[nx-1,:,:], grid[0,:,:],
    #                       grid[:,ny-1,:], grid[:,0,:],
    #                       grid[:,:,nz-1], grid[:,:,0]), axis = None)

    #exact slice
    face = np.concatenate((grid[nx-1,:,:], grid[0,:,:],
                           grid[1:nx-1,ny-1,:], grid[1:nx-1,0,:], 
                           grid[1:nx-1,1:ny-1,nz-1], grid[1:nx-1,1:ny-1,0]), axis = None)  

    
    return np.median(face)



def save_model(model, out_path, file_name):
    torch.save(model.state_dict(), out_path+"weights/"+file_name+".pth")
    with open(out_path + "model/"+ file_name+".pkl",'wb') as fp:
        pickle.dump(model, fp)

        
def load_model(out_path,file_name):
    with open(out_path + "model/" + file_name+".pkl", 'rb') as fp:
        model = pickle.load(fp)
    model.load_state_dict(torch.load(out_path+"weights/"+file_name+".pth"))
    return model


<<<<<<< HEAD


def create_bin(gfe_map):
    # Create bins using the GFE segregation
    '''
    0 for GFE above +RT. (I presume all “core” GFE values are above RT but it should be checked, of course. We don’t want any mislabelled voxel.)
    1 for GFE between 0 and +RT
    2 for GFE between -RT and 0
    3 for GFE between -2RT and -RT
    4 for GFE below -2RT
    '''


    return 0

def get_bin_frequency(gfe_map):
    # Return the frequency of GFE values in each binpossibly Variable-torch
    cp1= np.copy(gfe_map)
    cp2= np.copy(gfe_map)
    cp3= np.copy(gfe_map)
    cp4= np.copy(gfe_map)
    cp5= np.copy(gfe_map)

    cp1 = cp1[cp1>0.25*RT]
    cp2 = cp2[cp2>0]
    cp2 = cp2[cp2<0.25*RT]
    cp3 = cp3[cp3<0]
    cp3 = cp3[cp3>-0.25*RT]
    cp4 = cp4[cp4<-0.25*RT]
    cp4 = cp4[cp4>-0.5*RT]
    cp5 = cp5[cp5<-0.5*RT]

    freq_list = [len(cp1), len(cp2), len(cp3), len(cp4) ,len(cp5)]
    return freq_list

=======
>>>>>>> d445e6c7f65f5e126a27dc02e8b24120bad4d610
if __name__=='__main__':
    print("I need coffe! :( ")
