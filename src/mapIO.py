from __future__ import print_function
import numpy as np
import sys
import os
sys.path.append(os.path.join(os.path.dirname(__file__), ".."))
from src.util import pad_map, vec2grid

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

            if i > 5:  
                lines = line.split()
                dens.append(float(lines[0]))

            i = i + 1

    if i-6 != n_cells[0]*n_cells[1]*n_cells[2]:
        print("Error! Mismatch between provided and calculated volume size")
            

    #convert list to np.array
    dens = np.array(dens)

    #convert from 1D array to 3D array(tensor)
    dens = vec2grid(n_cells, dens) #ex-f-call
  
    return res, n_cells, dens
  
def get_target(map_path, map_names, pdb_id, batch, dim, cutoff = False):
    """
    This function invokes necessary frag maps, pads them
    and returns them with required tensor dimension.
    
    """
    map_tail = ".gfe.map"
    map_path_list = [map_path+pdb_id+"."+name+map_tail for name in map_names]


    n_batch = batch
    n_FM = 4
    map_tensor = np.zeros(shape = (n_batch, n_FM, dim, dim, dim))

    dmin = []
    dsize = []

    for i in range(len(map_path_list)):
        _, _, dens = read_map(map_path_list[i])      #f-call


        #apply min_max norm
        dmin.append(dens.min())
        dsize.append(dens.max() - dens.min())
        dens = (dens - dmin[i]) / (dsize[i])
        
        
        #apply cutoff
        #if cutoff == True:
        #    dens[dens > 0] = 0.0 
            
        #dens = np.abs(dens) # check!!!!!!!!!!!!!!!!!!!!!!
            
        pad_dens, xpad, ypad, zpad = pad_map(dens)   #ex-f-call
        map_tensor[n_batch-1, i,:,:,:] = pad_dens
       
    pad = [xpad,ypad,zpad]

    return map_tensor, pad, np.array(dmin), np.array(dsize)


def write_map(vec, out_path, out_name, ori, res, n):
    """
    This function outputs a .map file
    that is readable by vmd.
    """

    fname = out_path + out_name + ".gfe.map"
    nx = n[0]; ny = n[1]; nz = n[2]
    spacing = res      
    
    with open(fname,'w') as fp:

        fp.write("GRID_PARAMETER_FILE\n") 
        fp.write("GRID_DATA_FILE\n") 
        fp.write("MACROMOLECULE\n") 
        fp.write("SPACING %5.3f\n" % res)
        fp.write("NELEMENTS "+str(nx-1)+" "+str(ny-1)+ " "+str(nz-1)+"\n") 
        fp.write("CENTER "+str(ori[0])+" "+str(ori[1])+" "+str(ori[2])+"\n")

        for i in range(nx*ny*nz):
            fp.write("%10.3f\n" % vec[i])

            
if __name__=='__main__':

    import pyvista as pv

    frag_names = ["Benzene", "Propane", "H-bond Donor", "H-bond Acceptor"]
    path_list = ["../data/maps/1ycr.benc.gfe.map",
                 "../data/maps/1ycr.prpc.gfe.map",
                 "../data/maps/1ycr.hbacc.gfe.map",
                 "../data/maps/1ycr.hbdon.gfe.map"]
    chan_id = 2 # range 0-3

    ######------------- Test the read_map -----------------#######
    res, n_cells, dens = read_map(path_list[chan_id])
    print("Extracted volume dimention --> ",dens.shape)
    print("Specified dimension in the file header --> ", n_cells)

    #plot map density
    dens[dens > 0] = 0.0 #cutoff at zero!
    channel = dens
    p = pv.Plotter(point_smoothing = True)
    p.add_volume(np.abs(channel), cmap = "viridis", opacity = "linear")
    text = frag_names[chan_id]
    p.add_text(text, position = 'upper_left', font_size = 18)
    p.show()      
    

    ######------------- Testing write map ----------------#######
    out_path = "./"
    out_name = "test"
    ori = [40.250, -8.472, 20.406]
    res = 1.000
    n = [10,10,10]
    vec = 4*np.random.rand(n*n*n) - 2.0
    write_map(vec, out_path, out_name, ori, res, n)
    
