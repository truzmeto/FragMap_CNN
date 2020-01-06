import torch

def Rot90Seq(volume, iRot):
    """
    
    Input: volume - input tensor, dim = [batch_size, nchannels, dim, dim, dim] 
           iRot = [0,1,2,3....23] -  each unique 90 rotation

    Params:
           k    -- number of ratations, (k=0 -- no rotations, same as k=4)
           dims -- rotates from 1st int to 2nd int
                                     (i,j) = x --> y
                                     (i,k) = x --> z
                                     (j,k) = y --> z 

    Output:
          volume - input tensor, dim = [batch_size, nchannels, dim, dim, dim]
           
    """
    
    #store all integers that generate 24 rotations in a dictionary(hash tab)
    #for efficient table lookup. Also, last 3 indecies are always voxel indecies!

    
    if volume.nelement() == 0:
        raise ValueError("Volume tensor is empty!")

    i = len(volume.shape) - 3
    j = i + 1
    k = j + 1
    
                                                 #( x->y , y->z, x-->z) 
    Drot = {0:  [[0,i,j]],                           #(0,   0, 0) 
            1:  [[1,i,j]],                           #(90,  0, 0)
            2:  [[2,i,j]],                           #(180, 0, 0)
            3:  [[3,i,j]],                           #(270, 0, 0)
            #                                       
            4:  [[0,i,j], [1,j,k]],                  #(0,   90, 0) 
            5:  [[1,i,j], [1,j,k]],                  #(90,  90, 0)
            6:  [[2,i,j], [1,j,k]],                  #(180, 90, 0)
            7:  [[3,i,j], [1,j,k]],                  #(270, 90, 0)
            #                                       
            8:  [[0,i,j], [2,j,k]],                  #(0,   180, 0)
            9:  [[1,i,j], [2,j,k]],                  #(90,  180, 0)
            10: [[2,i,j], [2,j,k]],                  #(180, 180, 0)
            11: [[3,i,j], [2,j,k]],                  #(270, 100, 0)
            #          
            12: [[0,i,j], [3,j,k]],                  #(0,   270, 0)
            13: [[1,i,j], [3,j,k]],                  #(90,  270, 0)
            14: [[2,i,j], [3,j,k]],                  #(180, 270, 0)
            15: [[3,i,j], [3,j,k]],                  #(270, 270, 0)
            #                 
            16: [[0,i,j], [0,j,k], [1,i,k]],         #(0,   0, 90)
            17: [[1,i,j], [0,j,k], [1,i,k]],         #(90,  0, 90)
            18: [[2,i,j], [0,j,k], [1,i,k]],         #(180, 0, 90)
            19: [[3,i,j], [0,j,k], [1,i,k]],         #(270, 0, 90)
            #                            
            20: [[0,i,j], [0,j,k], [3,i,k]],         #(0,   0, 270)
            21: [[1,i,j], [0,j,k], [3,i,k]],         #(90,  0, 270)
            22: [[2,i,j], [0,j,k], [3,i,k]],         #(180, 0, 270)
            23: [[3,i,j], [0,j,k], [3,i,k]] }        #(270, 0, 270)

    
    rot_list = Drot[iRot]
    for subl in rot_list:
        volume = torch.rot90(volume, k = subl[0], dims = subl[1:3])
        
    return volume
    
