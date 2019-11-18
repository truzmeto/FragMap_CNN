import os
import sys
import torch.nn as nn
import torch.nn.functional as F
import torch
import numpy as np
    

def count_parameters(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


class CnnModel(nn.Module):
    def __init__(self, num_input_channels = 11):
        super(CnnModel, self).__init__()

        #to keep volume dims same
        s = 5
        p = (s-1)//2
        k_size = (s,s,s)
        pad = (p,p,p)
        
        self.conv = nn.Sequential(
            #conv layer 1
            nn.Conv3d(in_channels = num_input_channels,
                      out_channels = 8, #n convlolutions
                      kernel_size = k_size,
                      padding = pad),
	    nn.ReLU(),
            #conv layer 2
	    nn.Conv3d(in_channels = 8,
                      out_channels = 6,
                      kernel_size = k_size,
                      padding = pad),
            nn.ReLU(),
            #conv layer 3
            nn.Conv3d(in_channels = 6,
                      out_channels = 4,
                      kernel_size = k_size,
                      padding = pad),
            nn.ReLU()
            #nn.LeakyReLU()
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

    import pyvista as pv
    torch.manual_seed(3000)

    ########---------- Simple test with 1 forward pass -----------########
    c, d, h, w = 11, 20, 20, 20
    data = torch.randn(1, c, d, h, w)
    
    #invoke the model
    model = CnnModel()
    output = model(data)

    #compare input and output shapes
    print("First volume channel has the following dimentions")
    print(output[0,0,:,:,:].shape)

    #plot output density map
    chan_id = 0 # can be 0,1,2,3
    channel = output[0,chan_id,:,:,:].detach().numpy()
    p = pv.Plotter(point_smoothing=True)
    p.add_volume(channel, cmap="viridis", opacity="linear")
    p.show()

    
    
