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
        k_size = s #(s,s,s)
        pad = p #(p,p,p)
        
        self.conv = nn.Sequential(
            nn.Conv3d(in_channels = num_input_channels,
                      out_channels = 48, #n convlolutions
                      kernel_size = k_size,
                      padding = pad),
            #nn.BatchNorm3d(48),
            #nn.ReLU(),
            nn.LeakyReLU(),
            nn.MaxPool3d(kernel_size = k_size,
                         stride = (1,1,1),
                         padding = pad),
            nn.Conv3d(in_channels = 48,
                      out_channels = 24,
                      kernel_size = k_size,
                      padding = pad),
            #nn.BatchNorm3d(24),
            #nn.ReLU(),
            nn.LeakyReLU(),
            #nn.MaxPool3d(kernel_size = k_size,
            #             stride = (1,1,1),
            #             padding = pad),
            nn.Conv3d(in_channels = 24,
                      out_channels = 12,
                      kernel_size = k_size,
                      padding = pad),
            #nn.BatchNorm3d(12),
            #nn.ReLU(),
            #nn.LeakyReLU(),
            #nn.MaxPool3d(kernel_size = k_size,
            #             stride = (1,1,1),
            #             padding = pad),
            nn.Conv3d(in_channels = 12,
                      out_channels = 6,
                      kernel_size = k_size,
                      padding = pad),
            #nn.BatchNorm3d(6),
            #nn.ReLU()
            #nn.LeakyReLU()
        )
        
        
    def forward(self, input):
        #batch_size = input.size(0)
        conv_out = self.conv(input)
        return conv_out           


if __name__=='__main__':
    print("Enjoy life!") 
    
