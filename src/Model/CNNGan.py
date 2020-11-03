
"""
In progress
"""

import os
import sys
import torch.nn as nn
import torch.nn.functional as F
import torch
import numpy as np
  
    
    
#custom weigt init    
def weight_init(m):
    if isinstance(m, nn.Conv3d):
        size = m.weight.size()
        fan_out = size[1]*size[2]*size[3]*size[4] # number of rows
        fan_in = size[0]*size[2]*size[3]*size[4] # number of columns
        #variance = np.sqrt(2.0/(fan_out))
        variance = np.sqrt(4.0/(fan_out))
        m.weight.data.normal_(0.0, variance)
        m.bias.data.zero_()


    if isinstance(m, nn.Linear):
        size = m.weight.size()
        fan_out = size[0] # number of rows
        fan_in = size[1] # number of columns
        variance = np.sqrt(2.0/(fan_out))
        m.weight.data.normal_(0.0, variance)
        m.bias.data.zero_()

   

class Generator(nn.Module):
    def __init__(self, num_input_channels = 11):
        super(CnnGanModel, self).__init__()

        #to keep volume dims same
        k_size = 5 
        p = k_size // 2
        pad = p 
        
        self.conv = nn.Sequential(
            nn.Conv3d(in_channels = num_input_channels,
                      out_channels = 48, #n convlolutions
                      kernel_size = k_size,
                      padding = pad),
            #nn.BatchNorm3d(48),
            nn.LeakyReLU(),
            nn.MaxPool3d(kernel_size = k_size,
                         stride = (1,1,1),
                         padding = pad),
            nn.Conv3d(in_channels = 48,
                      out_channels = 24,
                      kernel_size = k_size,
                      padding = pad),
            #nn.BatchNorm3d(24),
            nn.LeakyReLU(),
            #nn.MaxPool3d(kernel_size = k_size,
            #             stride = (1,1,1),
            #             padding = pad),
            nn.Conv3d(in_channels = 24,
                      out_channels = 12,
                      kernel_size = k_size,
                      padding = pad),
            #nn.BatchNorm3d(12),
            #nn.LeakyReLU(),
            #nn.MaxPool3d(kernel_size = k_size,
            #             stride = (1,1,1),
            #             padding = pad),
            nn.Conv3d(in_channels = 12,
                      out_channels = 6,
                      kernel_size = k_size,
                      padding = pad),
        )
        
        
    def forward(self, input):
        conv_out = self.conv(input)
        return conv_out           





class Discriminator(nn.Module):
    def __init__(self):
        super(Discriminator, self).__init__()


         #to keep volume dims same
        k_size = 5 
        pad = k_size // 2
        
        self.conv1 = nn.Sequential(
            nn.Conv3d(in_channels = num_input_channels,
                      out_channels = 48, kernel_size = k_size, padding = pad),
            nn.LeakyReLU(),
            nn.MaxPool3d(kernel_size = k_size, stride = s, padding = pad),
            nn.Conv3d(in_channels = 48, out_channels = 24,
                      kernel_size = k_size, padding = pad),
            nn.LeakyReLU(),
            #nn.MaxPool3d(kernel_size = k_size,
            #             stride = s,
            #             padding = pad),
            nn.Conv3d(in_channels = 24,
                      out_channels = 12,
                      kernel_size = k_size,
                      padding = pad),
            #nn.LeakyReLU(),
            #nn.MaxPool3d(kernel_size = k_size,
            #             stride = 1,
            #             padding = pad),
            nn.Conv3d(in_channels = 12,
                      out_channels = 6,
                      kernel_size = k_size,
                      padding = pad),
        )
        
        

        
        self.full = nn.Sequential(
            nn.Linear(int(np.prod(img_shape)), 512),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Linear(512, 256),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Linear(256, 1),
            nn.Sigmoid(),
        )

    def forward(self, inpup, target):
        img_flat = img.view(img.size(0), -1)
        validity = self.model(img_flat)

        return validity


    
if __name__=='__main__':
    print("Enjoy life!") 
    
