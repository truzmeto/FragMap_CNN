import os
import sys
import torch.nn as nn
import torch.nn.functional as F
import torch
import numpy as np
  

def count_parameters(model):
    n_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    return n_params


class CnnModel(nn.Module):
    def __init__(self, num_input_channels = 11):
        super(CnnModel, self).__init__()

        #to keep volume dims same
        s = 3
        p = (s-1)//2
        k_size = (s,s,s)
        pad = (p,p,p)
        
        self.conv = nn.Sequential(
            #conv layer 1
            nn.Conv3d(in_channels = num_input_channels,
                      out_channels = 16, #n convlolutions
                      kernel_size = k_size,
                      padding = pad),
	    #nn.ReLU(),
            nn.LeakyReLU(),

            #conv layer 2
	    nn.Conv3d(in_channels = 16,
                      out_channels = 8,
                      kernel_size = k_size,
                      padding = pad),
            #nn.ReLU(),
            nn.LeakyReLU(),

            #conv layer 3
            nn.Conv3d(in_channels = 8,
                      out_channels = 4,
                      kernel_size = k_size,
                      padding = pad),
            #nn.ReLU()
            nn.LeakyReLU()
        )
        
        #self.fc = nn.Sequential(
	#    nn.Linear(512, 256),
	#    nn.ReLU(),
	#    nn.Linear(256, 128),
	#    nn.ReLU(),
	#    nn.Linear(128, 1)
        #)

        #self.apply(weight_init)
        
    def forward(self, input):
        #batch_size = input.size(0)
        conv_out = self.conv(input)
        return conv_out        
    

if __name__=='__main__':
    print("Enjoy life!")
    
    
