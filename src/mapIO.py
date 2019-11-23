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

    for i in range(len(map_path_list)):
        _, _, FrE = read_map(map_path_list[i])      #in-f-call

        
        #apply cutoff
        if cutoff == True:
            FrE[FrE > 0] = 0.0 
        
            
        #convert to density
        kBT = 0.6 #
        dens = np.exp(-FrE/kBT) 
            
        pad_dens, xpad, ypad, zpad = pad_map(dens)   #ex-f-call
        map_tensor[n_batch-1, i,:,:,:] = pad_dens
       
    pad = [xpad,ypad,zpad]

    return map_tensor, pad #, np.array(dmin), np.array(dsize)


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
    print("Patience is the key to everything!")
    
