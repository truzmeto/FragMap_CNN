import numpy as np
import random 
import sys  

def sample_batch(batch_size, pdb_ids, pdb_path, shuffle = True):
    """
    
    """      

    if batch_size > len(pdb_ids):
        print("Batch size must be less or equal to #pdbs")
        sys.exit(0)
        
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
    nx = n[0]; ny = n[1]; nz = n[2] 
    #grid = np.zeros(shape = (nz,ny,nx), dtype = float)
    grid = np.reshape(vec, newshape = (nx,ny,nz), order = "F")
    
    return grid


def grid2vec(n, grid):
    """
    This function transforms 3D grid.(nx,ny,nz) into
    vector (nx*ny*nz) 
    """
    nx = n[0]; ny = n[1]; nz = n[2] 
    #vec = np.zeros(shape = (nx*ny*nz), dtype = float)
    vec = np.reshape(grid, newshape = nx*ny*nz, order = "F")

    return vec


def pad_map(dens, maxD):
    """
    This function pads np.ndarray of size(nx,ny,nz)
    cubic box according to maxD provided. Zero padding
    applied at the end (or RHS) and returns
    np.ndarray of size(maxD,maxD,maxD).
    
    """
    
    dimx, dimy, dimz = dens.shape
    
    xpad = maxD - dimx
    ypad = maxD - dimy
    zpad = maxD - dimz
    
    #pad right hand side only
    dens = np.pad(dens,
                  pad_width = ((0,xpad), (0,ypad), (0,zpad)),
                  mode = 'constant') #zero padding by default
    #print(dens2.shape)
    return dens, xpad, ypad, zpad 


#def pad_map(dens):
#    """
#    This function converts np.ndarray of size(nx,ny,nz),
#    with unequal dimentions, adds a zero padding at the
#    end(or RHS) and returns np.ndarray of size(n,n,n),
#    where n is max dimention(max emong (nx,ny,nz))
#    
#    """
#    
#    dim = dens.shape # tuple
#    dimx, dimy, dimz = dim # unpack tuple
#
#    #apply padding if any pair of dimensions are non-equal
#    if dimx != dimy or dimx != dimz or dimy != dimz:  
#        
#        max_dim = np.array(dim).max() # get max dim
#        xpad = max_dim - dimx
#        ypad = max_dim - dimy
#        zpad = max_dim - dimz
#    
#        #pad right hand side only
#        dens = np.pad(dens,
#                      pad_width = ((0,xpad), (0,ypad), (0,zpad)),
#                      mode = 'constant') #zero padding by default
#        
#    return dens, xpad, ypad, zpad 


def unpad_map(dens, xpad, ypad, zpad):
    """
    This function unpads the volume by
    slicing out the original part

    """
    #unpad, remove padded parts
    n = dens.shape[0]
    dens = dens[0:n-xpad, 0:n-ypad, 0:n-zpad].copy()
   
    return dens


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
    n_face_vox = nx*ny*nz - (nx-2)*(ny-2)*(nz-2)
    face_ave = (face_x + face_y + face_z) / n_face_vox

    return face_ave


def test_model(model, saved_weights_path, output_path, map_names_list, pdb_ids, test_input):
    ori = [40.250, -8.472, 20.406] # TODO: Set Origin
    model.load_state_dict(torch.load(saved_weights_path))
    model.eval() #Needed to set into inference mode
    
    # TODO: Batch Test output
    for sample in range(len(test_input)):
        inp_vol = test_input[sample,:,:,:,:]
        output = model(inp_vol)
        for i in range(len(map_names_list)):
            out_name = pdb_ids[sample]+"."+ map_names_list[i]
            grid = output[0,i,:,:,:].cpu().detach().numpy()
            grid = unpad_map(grid, xpad = pad[0], ypad = pad[1], zpad = pad[2])

            #convert from Free-E to density 
            grid[grid <= 0.000] = 0.0001
            vol = grid #-kBT *np.log(grid)  

            #vol = grid*(gfe_max[i] - gfe_min[i]) + gfe_min[i] 
            #vol = grid
            nx, ny, nz = grid.shape
         
            vec = grid2vec([nx,ny,nz], vol)
            write_map(vec,
                      out_path,
                      out_name,
                      ori = ori,
                      res = resolution,
                      n = [nx,ny,nz])




if __name__=='__main__':
    print("Universal knowledge must be stored somewhere!")
