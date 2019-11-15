import numpy as np


def read_map(file_path):
    
    path = file_path
    
    with open(path) as file:
        i = 0
        dens = []
        for line in file:
            if i == 3:
                res = float(line.split()[1])
            elif i == 4:
                n_cells = [int(float(x))+1 for x in line.split()[1:]]  
            elif i == 5:
                center = [float(x) for x in line.split()[1:]]

            #print(i, line.split())
            if i > 5:  
                lines = line.split()
                dens.append(float(lines[0]))

            i = i + 1

    if i-6 != n_cells[0]*n_cells[1]*n_cells[2]:
        print("Error! Mismatch between provided and calculated volume size")
            

    #convert list to np.array
    dens = np.array(dens)

    #convert from 1D array to 3D array(tensor)
    dens = vec2grid(n_cells, dens)
  
    return res, n_cells, dens


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
    
def get_target(map_path, map_names, pdb_id, batch, dim):
    """

    """
    map_tail = ".gfe.map"
    map_path_list = [map_path+pdb_id+"."+name+map_tail for name in map_names]


    n_batch = batch
    n_FM = 4
    map_tensor = np.zeros(shape = (n_batch, n_FM, dim, dim, dim))
    for i in range(len(map_path_list)):
        _, _, dens = read_map(map_path_list[i])
        pad_dens = pad_map(dens)
        map_tensor[n_batch-1, i,:,:,:] = pad_dens

        
    return map_tensor

   
    
if __name__=='__main__':

    import pyvista as pv

    frag_names = ["Benzene", "Propane", "H-bond Donor", "H-bond Acceptor"]
    path_list = ["../data/maps/1ycr.benc.gfe.map",
                 "../data/maps/1ycr.prpc.gfe.map",
                 "../data/maps/1ycr.hbacc.gfe.map",
                 "../data/maps/1ycr.hbdon.gfe.map"]
    chan_id = 3 # range 0-3

    ######------------- Test the read_map -----------------#######
    res, n_cells, dens = read_map(path_list[chan_id])
    print("Extracted volume dimention --> ",dens.shape)
    print("Specified dimension if file header --> ", n_cells)

    #plot map density
    dens[dens > 0] = 0.0 #cutoff at zero!
    channel = dens
    p = pv.Plotter(point_smoothing = True)
    p.add_volume(np.abs(channel), cmap = "viridis", opacity = "linear")
    text = frag_names[chan_id]
    p.add_text(text, position='upper_left', font_size=18)
    p.show()      
    

    ######------------- Test padding ----------------#######
    pad_dens = pad_map(dens)
    if np.abs(pad_dens.sum() - dens.sum()) > 0.000001:
        print("Error! Zero padding should not affect the sum")
        print("Padded sum = ", pad_dens.sum())
        print("Map sum = ", dens.sum())
        
    else:
        print("Padding test passed!")

    

