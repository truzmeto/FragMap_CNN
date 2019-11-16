
def vec2grid(n, vec):
    """
    This function transforms 1D vec. into
    tensor (nx,ny,nz) data structure
    """
    nx = n[0]; ny = n[1]; nz = n[2] 
    grid = np.zeros(shape = (nz,ny,nx), dtype = float)
    grid = vec.reshape(nz,ny,nx) #order musb be inverted because
                                 #that is how map file is done :/
    return grid


def grid2vec(n, grid):
    """
    This function transforms 3D grid.(nx,ny,nz) into
    vector (nx*ny*nz) 
    """
    nx = n[0]; ny = n[1]; nz = n[2] 
    vec = np.zeros(nx*ny*nz), dtype = float)
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
    
    return dens
    
