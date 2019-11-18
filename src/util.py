import numpy as np

def vec2grid(n, vec):
    """
    This function transforms 1D vec. into
    tensor (nx,ny,nz) data structure
    """
    nx = n[0]; ny = n[1]; nz = n[2] 
    grid = np.zeros(shape = (nz,ny,nx), dtype = float)
    grid = vec.reshape(nz,ny,nx) #order must be inverted because
                                 #that is how map file is done :/
    return grid


def grid2vec(n, grid):
    """
    This function transforms 3D grid.(nx,ny,nz) into
    vector (nx*ny*nz) 
    """
    nx = n[0]; ny = n[1]; nz = n[2] 
    vec = np.zeros(shape =(nx*ny*nz), dtype = float)
    vec = grid.reshape(nx*ny*nz)

    return vec


def pad_map(dens):
    """
    This function converts np.ndarray of size(nx,ny,nz),
    with unequal dimentions, adds a zero padding at the
    end(or RHS) and returns np.ndarray of size(n,n,n),
    where n is max dimention(max emong (nx,ny,nz))
    
    """
    dim = dens.shape # tuple
    dimx, dimy, dimz = dim # unpack tuple

    #apply padding if any pair of dimensions are non-equal
    if dimx != dimy or dimx != dimz or dimy != dimz:  
        
        max_dim = np.array(dim).max() # get max dim
        xpad = max_dim - dimx
        ypad = max_dim - dimy
        zpad = max_dim - dimz
    
        #pad right hand side only
        dens = np.pad(dens,
                       pad_width = ((0,xpad),(0,ypad),(0,zpad)),
                       mode = 'constant') #zero padding by default
    
    return dens, xpad, ypad, zpad    #!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!

def unpad_map(dens, xpad, ypad, zpad):
    """
    This function unpadds the volume by
    slicing out the original part

    """
    #unpad the padded parts
    n = dens.shape[0]
    dens = dens[0:n-xpad, 0:n-ypad, 0:n-zpad].copy()
   
    return dens

if __name__=='__main__':

    import pyvista as pv
    from mapIO import read_map 

    frag_names = ["Benzene", "Propane", "H-bond Donor", "H-bond Acceptor"]
    path_list = ["../data/maps/1ycr.benc.gfe.map",
                 "../data/maps/1ycr.prpc.gfe.map",
                 "../data/maps/1ycr.hbacc.gfe.map",
                 "../data/maps/1ycr.hbdon.gfe.map"]
    chan_id = 1 # range 0-3
    _, _, dens = read_map(path_list[chan_id])
    
    
    ######------------- Test padding ----------------#######
    pad_dens, xpad, ypad, zpad = pad_map(dens)
    if np.abs(pad_dens.sum() - dens.sum()) > 0.000001:
        print("Error! Zero padding should not affect the sum")
        print("Padded sum = ", pad_dens.sum())
        print("Map sum = ", dens.sum())
        
    else:
        print("Padding test passed!")
        

    ######------------- Test unpadding ----------------#######
    ori_dim = dens.shape
    dens = unpad_map(pad_dens, xpad, ypad, zpad)
    unpad_dim = dens.shape
    i = 0; s=0
    for item in ori_dim:
        s = s + item - unpad_dim[i]
        i+=1

    if s != 0:
        print("Unpadding test failed!")
    else:
        print("Unpadding test passed!")
