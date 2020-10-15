from functools import partial
import torch

import sys
import os
sys.path.append(os.path.join(os.path.dirname(__file__), "../.."))

from e3nn import o3, rs
#from e3nn.image.convolution import Convolution 
from src.SE3Model.convolution import Convolution
from e3nn.image.filter import LowPassFilter
from e3nn.radial import GaussianRadialModel
from e3nn.tensor_product import LearnableTensorSquare

from e3nn.non_linearities.rescaled_act import sigmoid, swish, tanh
from e3nn.non_linearities.norm_activation import NormActivation


class E3nn(torch.nn.Module):
    def __init__(self, num_input_channels=11, out_channels=6):
        super(E3nn, self).__init__()

        size = 5
        pad = size // 2
        lmax = 1
        Rs = list(range(lmax + 1)) # [0,1,2,3]
        #Rs = [(2 * l + 1, l) for l in range(lmax + 1)]
        m = num_input_channels
        fuzzy_pixels = False
        #print(rs.dim(Rs))
        normalization = 'norm'

        self.conv = torch.nn.Sequential(
            Convolution(
                lmax = lmax,
                Rs_in = [(num_input_channels,0)],
                Rs_out = Rs*m, # m-scalars, m-vectors, rs.dim(Rs*m) = 44
                size = size,
                padding = pad, 
                fuzzy_pixels = fuzzy_pixels
            ),
           
            Convolution(
                lmax = lmax,
                Rs_in = Rs*m,
                Rs_out = Rs*m*4, # 2m-scalars, 2m-vectors, rs.dim(Rs*m*2) = 88
                size = size,
                padding = pad, 
                fuzzy_pixels = fuzzy_pixels
            ),
            NormActivation(Rs*m*4, sigmoid, normalization = normalization),
            
            Convolution(
                lmax = lmax,
                Rs_in = Rs*m*4,
                Rs_out = Rs*m*8, # 4m-scalars, 4m-vectors, rs.dim(Rs*m) = 176
                size = size,
                padding = pad, 
                fuzzy_pixels = fuzzy_pixels
            ),
            LowPassFilter(scale=2.0, stride=1),
                 
            Convolution(
                lmax = lmax,
                Rs_in = Rs*m*8, 
                Rs_out = Rs*m*4,
                size = size,
                padding = pad, 
                fuzzy_pixels = fuzzy_pixels
            ),
            LowPassFilter(scale=2.0, stride=1),
            
            Convolution(
                lmax = lmax,
                Rs_in = Rs*m*4,
                Rs_out = Rs*m,  
                size = size,
                padding = pad, 
                fuzzy_pixels = fuzzy_pixels
            ),
            NormActivation(Rs*m, sigmoid, normalization = normalization),

            
            Convolution(
                lmax = lmax,
                Rs_in = Rs*m, 
                Rs_out = [(out_channels, 0)], 
                size = size,
                padding = pad,
                fuzzy_pixels = fuzzy_pixels
            ),
        )

    def forward(self, input):
        conv_out = self.conv(input)
        return conv_out


