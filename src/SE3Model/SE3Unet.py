"""
SE(3) Unet model by T. Ruzmetov
"""

import sys
import os

import torch
import torch.nn as nn
import torch.nn.functional as F

from e3nn.non_linearities.rescaled_act import sigmoid, swish, tanh
from e3nn.non_linearities.norm_activation import NormActivation
from e3nn.batchnorm import BatchNorm

sys.path.append(os.path.join(os.path.dirname(__file__), "../.."))
from src.SE3Model.convolution import Convolution


def ConvBlock1(Rs_in, Rs_out, lmax, size, fpix):
    return nn.Sequential(
        Convolution(Rs_in, Rs_out, lmax=lmax, size = size, stride = 1, padding = size//2, fuzzy_pixels = fpix),
        #BatchNorm(Rs_out, normalization='component'),
        NormActivation(Rs_out, swish, normalization = 'component'),
    )

   
def ConvBlock2(Rs_in, Rs_out, lmax, size, fpix, stride):
    return nn.Sequential(
        ConvBlock1(Rs_in, Rs_out, lmax, size, fpix),
        Convolution(Rs_out, Rs_out, lmax=lmax,  size=size, stride=stride, padding = size//2, fuzzy_pixels = fpix),
        #BatchNorm(Rs_out, normalization='component'), 
    )


def ConvTransBlock(Rs_in, Rs_out, lmax, size, fpix):
    return nn.Sequential(
        Convolution(Rs_in, Rs_out, lmax=lmax, size=size, stride=2, padding=size//2, output_padding=1, transpose=True, fuzzy_pixels = fpix),
        #BatchNorm(Rs_out, normalization='component'), 
        NormActivation(Rs_out, swish, normalization = 'componenet'),
    )



class UNet(nn.Module):
    def __init__(self, size, mult, lmax, inp_channels=11, out_channels=6):
        super(UNet, self).__init__()
        
        Rs_in = [(inp_channels,0)]
        Rs_out =  [(out_channels,0)]
        m = mult #multiplier
        Rs = list(range(lmax + 1)) #base Rs
        fp = False  #option to add noise to conv kernels
        st = 2     #downlampling stride
        
        # Down sampling
        self.down_1 = ConvBlock2(Rs_in,      m *      Rs, lmax=lmax, size=size, fpix=fp, stride=st)
        self.down_2 = ConvBlock2(m *     Rs, m * 2  * Rs, lmax=lmax, size=size, fpix=fp, stride=st)
        self.down_3 = ConvBlock2(m * 2 * Rs, m * 4  * Rs, lmax=lmax, size=size, fpix=fp, stride=st)
        self.down_4 = ConvBlock2(m * 4 * Rs, m * 8  * Rs, lmax=lmax, size=size, fpix=fp, stride=st) 
        self.down_5 = ConvBlock2(m * 8 * Rs, m * 16 * Rs, lmax=lmax, size=size, fpix=fp, stride=st)
        
        # Bridge
        self.bridge = ConvBlock2(m *16 * Rs, m * 32 * Rs, lmax=lmax, size=size, fpix=fp, stride=1)        
                
        # Up sampling
        self.trans_1 = ConvTransBlock(m * 32 * Rs, m * 32 * Rs, lmax=lmax, size=size, fpix=fp)
        self.up_1    = ConvBlock2(    m * 48 * Rs, m * 16 * Rs, lmax=lmax, size=size, fpix=fp, stride=1)
        self.trans_2 = ConvTransBlock(m * 16 * Rs, m * 16 * Rs, lmax=lmax, size=size, fpix=fp )
        self.up_2    = ConvBlock2(    m * 24 * Rs, m * 8  * Rs, lmax=lmax, size=size, fpix=fp, stride=1)
        self.trans_3 = ConvTransBlock(m * 8  * Rs, m * 8  * Rs, lmax=lmax, size=size, fpix=fp)
        self.up_3    = ConvBlock2(    m * 12 * Rs, m * 4  * Rs, lmax=lmax, size=size, fpix=fp, stride=1)
        self.trans_4 = ConvTransBlock(m * 4  * Rs, m * 4  * Rs, lmax=lmax, size=size, fpix=fp)
        self.up_4    = ConvBlock2(    m * 6  * Rs, m * 2  * Rs, lmax=lmax, size=size, fpix=fp, stride=1)
        self.trans_5 = ConvTransBlock(m * 2  * Rs, m * 2  * Rs, lmax=lmax, size=size, fpix=fp)
        self.up_5    = ConvBlock2(    m * 3  * Rs, m * 1  * Rs, lmax=lmax, size=size, fpix=fp, stride=1)
        
        # Output
        self.out = ConvBlock1(m * Rs, Rs_out, lmax=lmax, size=size, fpix=fp)
        #self.out = ConvBlock1(m * Rs, Rs_out, lmax=lmax, size=size, fpix=fp)

    
    def skip(self, uped, bypass):
        """
        This functions matches size between bypass and upsampled feature.
        It pads or unpads bypass depending on how different dims are.
        """

        uped = torch.einsum('txyzi->tixyz', uped)
        bypass = torch.einsum('txyzi->tixyz', bypass)
        p = bypass.shape[2] - uped.shape[2]                                                                                          
    
        if p == 0:
            out = torch.cat((uped, bypass), 1)
            out = torch.einsum('tixyz->txyzi', out)
            return out
        else:
            pl = p // 2
            pr = p - pl
            bypass = F.pad(bypass, (-pl, -pr, -pl, -pr, -pl, -pr))
            out = torch.cat((uped, bypass), 1)
            out = torch.einsum('tixyz->txyzi', out)
            
            return out
    
        
    def forward(self, x):
        # Down sampling
        down_1 = self.down_1(x) 
        #print("down1", down_1.shape)
        
        down_2 = self.down_2(down_1) 
        #print("down2", down_2.shape)
        
        down_3 = self.down_3(down_2) 
        #print("down3", down_3.shape) 
       
        down_4 = self.down_4(down_3) 
        #print("down4", down_4.shape)
        
        down_5 = self.down_5(down_4) 
        #print("down5", down_5.shape)
       
        #print('----------------------------')
        
        # Bridge
        bridge = self.bridge(down_5) 

        #print("----------------------------")
        
        # Up sampling
        trans_1 = self.trans_1(bridge) 
        concat_1 = self.skip(trans_1, down_5)
        up_1 = self.up_1(concat_1) 
        #print("trans1", trans_1.shape) 
        #print("concat1", concat_1.shape)
        #print('up1', up_1.shape)         
                       
        trans_2 = self.trans_2(up_1) 
        concat_2 = self.skip(trans_2, down_4)
        up_2 = self.up_2(concat_2)
        #print("trans2", trans_2.shape)  
        #print("concat2", concat_2.shape)
        #print('up2', up_2.shape)        

        
        trans_3 = self.trans_3(up_2) 
        concat_3 = self.skip(trans_3, down_3)
        up_3 = self.up_3(concat_3)  
        #print("trans3", trans_3.shape)  
        #print("concat3", concat_3.shape)
        #print('up3', up_3.shape)
        
        
        trans_4 = self.trans_4(up_3) 
        concat_4 = self.skip(trans_4, down_2)
        up_4 = self.up_4(concat_4) 
        #print("trans4", trans_4.shape)  
        #print("concat4", concat_4.shape)
        #print('up4', up_4.shape)        

        
        trans_5 = self.trans_5(up_4) 
        concat_5 = self.skip(trans_5, down_1)
        up_5 = self.up_5(concat_5) 
        #print("trans5", trans_5.shape)  
        #print("concat5", concat_5.shape)
        #print('up5', up_5.shape)        
        
        # Output
        out = self.out(up_5)
        
        return out

    
if __name__ == "__main__":
  device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

  inp_size = 64
  inp_channels = 1
  out_channels = 1

  lmax = 1
  k_size = 3
  m = 2 #multiplier
  
  x = torch.Tensor(1, inp_size, inp_size, inp_size, inp_channels)
  x.to(device)
  print("x size: {}".format(x.size()))
  
  model = UNet(size = k_size, mult = m, lmax = lmax, inp_channels = inp_channels, out_channels = out_channels)
  
  out = model(x)
  print("out size: {}".format(out.size()))
