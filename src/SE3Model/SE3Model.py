from functools import partial
import torch

import sys
import os

from e3nn import o3, rs
from e3nn.image.filter import LowPassFilter
from e3nn.radial import GaussianRadialModel
from e3nn.tensor_product import LearnableTensorSquare
from e3nn.non_linearities.rescaled_act import sigmoid, swish, tanh
from e3nn.non_linearities.norm_activation import NormActivation


sys.path.append(os.path.join(os.path.dirname(__file__), "../.."))
from src.SE3Model.convolution import Convolution


class E3nn(torch.nn.Module):
    def __init__(self, num_input_channels=11, out_channels=6):
        super(E3nn, self).__init__()

        size = 7
        pad = size // 2
        lmax = 2
        Rs = list(range(lmax + 1)) # [0,1,2]
        m = 4 #multiplier
        fuzzy_pixels = True
        #print(rs.dim(Rs))
        normalization = 'component'

        self.conv = torch.nn.Sequential(
            Convolution(
                lmax = lmax,
                Rs_in = [(num_input_channels,0)],
                Rs_out = Rs*m, # m-scalars, m-vectors of l=1, m-vectors of l=2, rs.dim(Rs*m) = 36
                size = size,
                padding = pad, 
                fuzzy_pixels = fuzzy_pixels
            ),
           
            Convolution(
                lmax = lmax,
                Rs_in = Rs*m,
                Rs_out = Rs*m*4, # 4m-scalars, 4m-vectors of l=1, 4m-vectors of l=2, rs.dim(Rs*m*4) = 144
                size = size,
                padding = pad, 
                fuzzy_pixels = fuzzy_pixels
            ),
            NormActivation(Rs*m*4, tanh, normalization = normalization),
            
            Convolution(
                lmax = lmax,
                Rs_in = Rs*m*4,
                Rs_out = Rs*m*8, # 8m-scalars, 8m-vectors of l=1, 8m-vectors of l=2, rs.dim(Rs*m) = 288
                size = size,
                padding = pad, 
                fuzzy_pixels = fuzzy_pixels
            ),
            #LowPassFilter(scale=2.0, stride=1), #needed befor downsampling!
                 
            Convolution(
                lmax = lmax,
                Rs_in = Rs*m*8, 
                Rs_out = Rs*m*4,
                size = size,
                padding = pad, 
                fuzzy_pixels = fuzzy_pixels
            ),
            #LowPassFilter(scale=2.0, stride=1),
            NormActivation(Rs*m*4, tanh, normalization = normalization),

            Convolution(
                lmax = lmax,
                Rs_in = Rs*m*4,
                Rs_out = Rs*m,  
                size = size,
                padding = pad, 
                fuzzy_pixels = fuzzy_pixels
            ),
            NormActivation(Rs*m, tanh, normalization = normalization),
            
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


