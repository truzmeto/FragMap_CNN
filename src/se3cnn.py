import os
import sys 
import torch
from torch import nn
from torch.nn.modules.module import Module
import torch.nn.functional as F
import numpy as np

from se3cnn import SE3Convolution
from se3cnn.non_linearities import ScalarActivation
from .MultiplyVolumes import MultiplyVolumes


class SE3CnnModel(nn.Module):
    def __init__(self, num_input_channels = 11):
        super(SE3CnnModel, self).__init__()

        #to keep volume dims same
        #s = 3
        #p = (s-1)//2
        #k_size = s #(s,s,s)
        #pad = p #(p,p,p)
        # iN PROGRESS!
        	
	self.num_ch_1 = multiplier*2
	self.num_ch_2 = multiplier*4
        self.conv = nn.Sequential(
	    SE3Convolution([(num_input_channels,0)], [(multiplier*2,0)], size=5, dyn_iso=True, padding=2, bias=None),
	    nn.ReLU(),
            # ScalarActivation([(multiplier*2, F.relu)], bias=False),
	    
	    SE3Convolution([(multiplier*2,0)], [(multiplier*2,0)], size=5, dyn_iso=True, padding=2, bias=None),
            nn.ReLU(),
	    
	    SE3Convolution( [(multiplier*2,0)], [(multiplier*2,0)], size=5, dyn_iso=True, padding=2, bias=None),
            nn.ReLU(),
	    		
        )

        
        
    def forward(self, input):
        conv_out = self.conv(input)
        return conv_out           


#custom weigt init    
def init_weights(m):
	if type(m) == nn.Conv3d:
		torch.nn.init.xavier_uniform_(m.weight)
	if type(m) == nn.Linear:
		torch.nn.init.xavier_uniform_(m.weight)

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


        
    
def count_parameters(model):
    n_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    return n_params



class MinPool3d(nn.MaxPool3d):
    """
    This performe min-poling, the exact opposite of max-pooling.
    It is useful when prediction aims negative valued data points.
    """
    def forward(self, input):
        return -F.max_pool3d(-input, self.kernel_size, self.stride,
                            self.padding, self.dilation, self.ceil_mode,
                            self.return_indices)
    
if __name__=='__main__':
    print("Enjoy life!") 
    
