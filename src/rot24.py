import numpy as np
import random 
import sys  
import torch


def Rot90Seq(volume, iRot):
    """
    
    Input: volume - input tensor, dim = [batch_size, nchannels, dim, dim, dim] 
           iRot = [0,1,2,3....23] -  each unique 90 rotation

    Params:
           k    -- number of ratations, (k=0 -- no rotations, same as k=4)
           dims -- rotates from 1st to 2nd
                                     (2,3) = x --> y
                                     (2,4) = x --> z
                                     (3,4) = y --> z 

    Output:
          volume - input tensor, dim = [batch_size, nchannels, dim, dim, dim]
           
    """
    
    #store all integers that generate 24 rotations in a dictionary(hash tab)
    #for efficient tab. lookup
  
                                                  #( x->y , y->z, x-->z) 
    Drot = {0:  [[0,2,3]],                           #(0,   0, 0) 
            1:  [[1,2,3]],                           #(90,  0, 0)
            2:  [[2,2,3]],                           #(180, 0, 0)
            3:  [[3,2,3]],                           #(270, 0, 0)
            #                                        
            4:  [[0,2,3], [1,3,4]],                  #(0,   90, 0) 
            5:  [[1,2,3], [1,3,4]],                  #(90,  90, 0)
            6:  [[2,2,3], [1,3,4]],                  #(180, 90, 0)
            7:  [[3,2,3], [1,3,4]],                  #(270, 90, 0)
            #                                         
            8:  [[0,2,3], [2,3,4]],                  #(0,   180, 0)
            9:  [[1,2,3], [2,3,4]],                  #(90,  180, 0)
            10: [[2,2,3], [2,3,4]],                  #(180, 180, 0)
            11: [[3,2,3], [2,3,4]],                  #(270, 100, 0)
            #
            12: [[0,2,3], [3,3,4]],                  #(0,   270, 0)
            13: [[1,2,3], [3,3,4]],                  #(90,  270, 0)
            14: [[2,2,3], [3,3,4]],                  #(180, 270, 0)
            15: [[3,2,3], [3,3,4]],                  #(270, 270, 0)
            #
            16: [[0,2,3], [0,3,4], [1,2,4]],         #(0,   0, 90)
            17: [[1,2,3], [0,3,4], [1,2,4]],         #(90,  0, 90)
            18: [[2,2,3], [0,3,4], [1,2,4]],         #(180, 0, 90)
            19: [[3,2,3], [0,3,4], [1,2,4]],         #(270, 0, 90)
            #                          
            20: [[0,2,3], [0,3,4], [3,2,4]],         #(0,   90, 180)
            21: [[1,2,3], [0,3,4], [3,2,4]],         #(90,  90, 180)
            22: [[2,2,3], [0,3,4], [3,2,4]],         #(180, 90, 180)
            23: [[3,2,3], [0,3,4], [3,2,4]] }        #(270, 90, 180)

    
    rot_list = Drot[iRot]
    for subl in rot_list:
        volume = torch.rot90(volume, k = subl[0], dims = subl[1:3])

    return volume