"""
This code works with old SE3 repository
"""

import os
import sys
import torch
from torch import nn
from torch.nn.modules.module import Module
import torch.nn.functional as F
import numpy as np

#from se3cnn.image.convolution import SE3Convolution
from se3cnn import SE3Convolution


class FragMapSE3(Module):
    def __init__(self, inp_chans=11):
        super(FragMapSE3, self).__init__()

        #to keep volume dims same
        k = 5
        p = (k-1)//2

        self.conv = nn.Sequential(
            SE3Convolution([(inp_chans,0)], [(32,1)], size=k, dyn_iso=True, padding=p, bias=None),
            nn.ReLU(),
                      
            SE3Convolution([(32,1)], [(64,1)], size=k, dyn_iso=True, padding=p, bias=None),
            nn.ReLU(),
            #MinPool3d(kernel_size = k,
            #             stride = (1,1,1),
            #             padding = p),


            SE3Convolution( [(64,1)], [(32, 0)], size=k, dyn_iso=True, padding=p, bias=None),
            nn.ReLU(),
            nn.Dropout3d(0.1),
            
            SE3Convolution( [(32,0)],  [(6,0)], size=k, dyn_iso=True, padding=p, bias=None),
        )

        #self.apply(init_weights)


    def forward(self, input):
        conv_out = self.conv(input)
        return conv_out


#custom weigt init
def init_weights(m):
    print(type(m))
    if type(m) == nn.Conv3d:
        torch.nn.init.xavier_uniform_(m.weight)
    if type(m) == nn.Linear:
        torch.nn.init.xavier_uniform_(m.weight)



def count_parameters(model):
    n_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    return n_params


class MinPool3d(nn.MaxPool3d):
    """

    """

    def forward(self, input):
        return -F.max_pool3d(-input, self.kernel_size, self.stride,
                             self.padding, self.dilation, self.ceil_mode,
                             self.return_indices)



if __name__=='__main__':
    print("dude")
