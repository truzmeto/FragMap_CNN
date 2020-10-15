import numpy as np
import sys  
import os
import torch
sys.path.append(os.path.join(os.path.dirname(__file__), ".."))


def get_random_rotation(volume, target):
    '''
    Input: tensor
    Output: Randomly rotated tensor


    Note:
    Where is the evidence that this generates all 24 rotations?
    '''
    
    for i in range(3):
        dims = np.random.choice([2,3,4], 2, replace = False)
        k = np.random.choice([1,2,3])
        rot_volume = torch.rot90(volume, k = int(k), dims = list(dims))
        rot_target = torch.rot90(target, k = int(k), dims = list(dims))
        
    return rot_volume, rot_target
