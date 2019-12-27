import numpy as np
import sys
import os

sys.path.append(os.path.join(os.path.dirname(__file__), ".."))
from src.util import vec2grid

def read_map(file_path):
    
    path = file_path
    with open(path) as file:
        i = 0
        gfe = []
        for line in file:

            if i == 3:
                res = float(line.split()[1])
            elif i == 4:
                n_cells = [int(float(x))+1 for x in line.split()[1:]]  
            elif i == 5:
                center = [float(x) for x in line.split()[1:]]

            if i > 5:  
                lines = line.split()
                gfe.append(float(lines[0]))

            i = i + 1

    if i-6 != n_cells[0]*n_cells[1]*n_cells[2]:
        raise ValueError("Mismatch between provided and calc. #voxels") 
        

    #convert list to np.array
    gfe = np.array(gfe)
    
    #convert from 1D array to 3D array(tensor)
    gfe = vec2grid(n_cells, gfe) #ex-f-call
    
    return res, n_cells, gfe, center


def greatest_dim(target_path, target_list):
    """
    Input: 
    target_path: Path to fragment maps
    target_list: list of samples in the batch
    
    """

    max_dim = 0
    for targets in target_list:
        path = target_path + targets + ".benc.gfe.map" 
        with open(path) as file:
            t = file.readlines()[4].split(" ")
            for i in t[1:]:
                if max_dim < int(i):
                    max_dim = int(i)

    return max_dim


def write_map(vec, out_path, out_name, ori, res, n):
    """
    This function outputs a .map file
    that is readable by vmd.
    """

    fname = out_path + out_name + ".gfe.map"
    nx, ny, nz = n
    spacing = res      
    
    with open(fname,'w') as fp:

        fp.write("GRID_PARAMETER_FILE\n") 
        fp.write("GRID_DATA_FILE\n") 
        fp.write("MACROMOLECULE\n") 
        fp.write("SPACING %5.3f\n" % res)
        fp.write("NELEMENTS "+str(nx-1)+" "+str(ny-1)+ " "+str(nz-1)+"\n") #consider changing!!!!! 
        fp.write("CENTER "+str(ori[0])+" "+str(ori[1])+" "+str(ori[2])+"\n")

        for i in range(nx*ny*nz):
            fp.write("%10.3f\n" % vec[i])

            
if __name__=='__main__':
    print(" ^^__O__ ^^ ")
    
