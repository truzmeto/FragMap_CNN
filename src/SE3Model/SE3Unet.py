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


sys.path.append(os.path.join(os.path.dirname(__file__), "../.."))
from src.SE3Model.convolution import Convolution

def conv_block_3d(Rs_in, Rs_out):
    return nn.Sequential(
        Convolution(Rs_in, Rs_out, lmax=0,  size = 3, stride = 1, padding = 1, fuzzy_pixels = True),
        #nn.BatchNorm3d(out_dim),
        NormActivation(Rs_out, tanh, normalization = 'component'),
    )


def conv_trans_block_3d(Rs_in, Rs_out):
    return nn.Sequential(
        Convolution(Rs_in, Rs_out, lmax=0, size=3, stride=2, padding=1, output_padding=1, transpose=True, fuzzy_pixels = True),
        #nn.BatchNorm3d(out_dim),
        NormActivation(Rs_out, tanh, normalization = 'componenet'),
    )


class MPool3D(nn.MaxPool3d):
    """
    This performes max-poling
    """
    def forward(self, input):
        input = torch.einsum('txyzi->tixyz', input)
        output = F.max_pool3d(input, self.kernel_size, self.stride, self.padding,
                              self.dilation, self.ceil_mode, self.return_indices)
        output = torch.einsum('tixyz->txyzi', output)
        return output

    
def max_pooling_3d():
    return MPool3D(kernel_size=2, stride=2, padding=0)


    
def conv_block_2_3d(Rs_in, Rs_out):
    return nn.Sequential(
        conv_block_3d(Rs_in, Rs_out),
        Convolution(Rs_out, Rs_out, lmax=0,  size = 3, stride = 1, padding = 1, fuzzy_pixels = True),
        #nn.BatchNorm3d(out_dim),
    )



class UNet(nn.Module):
    def __init__(self, Rs, inp_channels=11, out_channels=6):
        super(UNet, self).__init__()
        
        Rs_in = [(inp_channels,0)]
        Rs_out =  [(out_channels,0)]
        m = 4 #multiplier
        
        # Down sampling
        self.down_1 = conv_block_2_3d(Rs_in, m * Rs)
        self.pool_1 = max_pooling_3d()
        self.down_2 = conv_block_2_3d(m * Rs, m * 2 * Rs)
        self.pool_2 = max_pooling_3d()
        self.down_3 = conv_block_2_3d(m * 2 * Rs, m * 4 * Rs)
        self.pool_3 = max_pooling_3d()
        self.down_4 = conv_block_2_3d(m * 4 * Rs, m * 8 * Rs) 
        self.pool_4 = max_pooling_3d()
        self.down_5 = conv_block_2_3d(m * 8 * Rs, m * 16 * Rs)
        self.pool_5 = max_pooling_3d()
        
        # Bridge
        self.bridge = conv_block_2_3d(m * 16 * Rs, m * 32 * Rs)
        
        # Up sampling
        self.trans_1 = conv_trans_block_3d(m * 32 * Rs, m * 32 * Rs)
        self.up_1 = conv_block_2_3d(m * 48 * Rs, m * 16 * Rs)
        self.trans_2 = conv_trans_block_3d(m * 16 * Rs, m * 16 * Rs)
        self.up_2 = conv_block_2_3d(m * 24 * Rs, m * 8 * Rs)
        self.trans_3 = conv_trans_block_3d(m * 8 * Rs, m * 8 * Rs)
        self.up_3 = conv_block_2_3d(m * 12 * Rs, m * 4 * Rs)
        self.trans_4 = conv_trans_block_3d(m * 4 * Rs, m * 4 * Rs)
        self.up_4 = conv_block_2_3d(m * 6 * Rs, m * 2 * Rs)
        self.trans_5 = conv_trans_block_3d(m * 2 * Rs, m * 2 * Rs)
        self.up_5 = conv_block_2_3d(m * 3 * Rs, m * 1 * Rs)
        
        # Output
        self.out = conv_block_3d(m * Rs, Rs_out)

    
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
        pool_1 = self.pool_1(down_1) 
        #print("pool1", pool_1.shape)
        
        down_2 = self.down_2(pool_1) 
        #print("down2", down_2.shape)
        pool_2 = self.pool_2(down_2) 
        #print("pool2", pool_2.shape)
        
        down_3 = self.down_3(pool_2) 
        pool_3 = self.pool_3(down_3) 
        #print("down3", down_3.shape) 
        #print("pool3", pool_3.shape)

        down_4 = self.down_4(pool_3) 
        pool_4 = self.pool_4(down_4)  
        #print("down4", down_4.shape)
        #print("pool4", pool_4.shape)
        
        down_5 = self.down_5(pool_4) 
        pool_5 = self.pool_5(down_5) 
        #print("down5", down_5.shape)
        #print("pool5", pool_5.shape)

        #print('----------------------------')
        
        # Bridge
        bridge = self.bridge(pool_5) 
        #print(bridge.shape)

        #print("----------------------------")
        
        # Up sampling
        trans_1 = self.trans_1(bridge) 
        #print("trans1", trans_1.shape) 
        concat_1 = self.skip(trans_1, down_5)
        up_1 = self.up_1(concat_1) 
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

  inp_size = 96
  in_channels = 11
  out_channels = 6

  lmax = 0
  Rs = list(range(lmax + 1)) 
  num_filters = 4 #multiplier
  fuzzy_pixels = True
  
  x = torch.Tensor(1, inp_size, inp_size, inp_size, in_channels)
  x.to(device)
  print("x size: {}".format(x.size()))
  
  model = UNet(Rs, inp_channels = 11, out_channels = 6)
  
  out = model(x)
  print("out size: {}".format(out.size()))
