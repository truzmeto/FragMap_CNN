# 3D-UNet model by UdonDa @ https://github.com/UdonDa/3D-UNet-PyTorch
# Model is adapted from Udonda
import torch
import torch.nn as nn
import torch.nn.functional as F

def conv_block_3d(in_dim, out_dim, activation):
    return nn.Sequential(
        nn.Conv3d(in_dim, out_dim, kernel_size=3, stride=1, padding=1),
        nn.BatchNorm3d(out_dim),
        activation,)


def conv_trans_block_3d(in_dim, out_dim, activation):
    return nn.Sequential(
        nn.ConvTranspose3d(in_dim, out_dim, kernel_size=3, stride=2, padding=1, output_padding=1),
        nn.BatchNorm3d(out_dim),
        activation,)


def max_pooling_3d():
    return nn.MaxPool3d(kernel_size=2, stride=2, padding=0)


def conv_block_2_3d(in_dim, out_dim, activation):
    return nn.Sequential(
        conv_block_3d(in_dim, out_dim, activation),
        nn.Conv3d(out_dim, out_dim, kernel_size=3, stride=1, padding=1),
        nn.BatchNorm3d(out_dim),)



class UNet(nn.Module):
    def __init__(self, in_dim, out_dim, num_filters):
        super(UNet, self).__init__()
        
        self.in_dim = in_dim
        self.out_dim = out_dim
        self.num_filters = num_filters
        activation = nn.LeakyReLU(0.2, inplace=True)
        
        # Down sampling
        self.down_1 = conv_block_2_3d(self.in_dim, self.num_filters, activation)
        self.pool_1 = max_pooling_3d()
        self.down_2 = conv_block_2_3d(self.num_filters, self.num_filters * 2, activation)
        self.pool_2 = max_pooling_3d()
        self.down_3 = conv_block_2_3d(self.num_filters * 2, self.num_filters * 4, activation)
        self.pool_3 = max_pooling_3d()
        self.down_4 = conv_block_2_3d(self.num_filters * 4, self.num_filters * 8, activation)
        self.pool_4 = max_pooling_3d()
        self.down_5 = conv_block_2_3d(self.num_filters * 8, self.num_filters * 16, activation)
        self.pool_5 = max_pooling_3d()
        
        # Bridge
        self.bridge = conv_block_2_3d(self.num_filters * 16, self.num_filters * 32, activation)
        
        # Up sampling
        self.trans_1 = conv_trans_block_3d(self.num_filters * 32, self.num_filters * 32, activation)
        self.up_1 = conv_block_2_3d(self.num_filters * 48, self.num_filters * 16, activation)
        self.trans_2 = conv_trans_block_3d(self.num_filters * 16, self.num_filters * 16, activation)
        self.up_2 = conv_block_2_3d(self.num_filters * 24, self.num_filters * 8, activation)
        self.trans_3 = conv_trans_block_3d(self.num_filters * 8, self.num_filters * 8, activation)
        self.up_3 = conv_block_2_3d(self.num_filters * 12, self.num_filters * 4, activation)
        self.trans_4 = conv_trans_block_3d(self.num_filters * 4, self.num_filters * 4, activation)
        self.up_4 = conv_block_2_3d(self.num_filters * 6, self.num_filters * 2, activation)
        self.trans_5 = conv_trans_block_3d(self.num_filters * 2, self.num_filters * 2, activation)
        self.up_5 = conv_block_2_3d(self.num_filters * 3, self.num_filters * 1, activation)
        
        # Output
        self.out = conv_block_3d(self.num_filters, out_dim, activation)

    
    def skip(self, uped, bypass):
        """
        This functions matches size between bypass and upsampled feature.
        It pads or unpads bypass depending on how different dims are.
        """
        
        p = bypass.shape[2] - uped.shape[2]                                                                                          
    
        if p == 0:
            return torch.cat((uped, bypass), 1)
        else:
            pl = p // 2
            pr = p - pl
            bypass = F.pad(bypass, (-pl, -pr, -pl, -pr, -pl, -pr))
            
        return torch.cat((uped, bypass), 1)
        

        
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
        #concat_1 = torch.cat([trans_1, down_5], dim=1) 
        concat_1 = self.skip(trans_1, down_5)
        up_1 = self.up_1(concat_1) 
        #print("concat1", concat_1.shape)
        #print('up1', up_1.shape)         
                       
        trans_2 = self.trans_2(up_1) 
        #concat_2 = torch.cat([trans_2, down_4], dim=1) 
        concat_2 = self.skip(trans_2, down_4)
        up_2 = self.up_2(concat_2)
        #print("trans2", trans_2.shape)  
        #print("concat2", concat_2.shape)
        #print('up2', up_2.shape)        

        
        trans_3 = self.trans_3(up_2) 
        #concat_3 = torch.cat([trans_3, down_3], dim=1) 
        concat_3 = self.skip(trans_3, down_3)
        up_3 = self.up_3(concat_3)  
        #print("trans3", trans_3.shape)  
        #print("concat3", concat_3.shape)
        #print('up3', up_3.shape)
        
        
        trans_4 = self.trans_4(up_3) 
        #concat_4 = torch.cat([trans_4, down_2], dim=1) 
        concat_4 = self.skip(trans_4, down_2)
        up_4 = self.up_4(concat_4) 
        #print("trans4", trans_4.shape)  
        #print("concat4", concat_4.shape)
        #print('up4', up_4.shape)        

        
        trans_5 = self.trans_5(up_4) 
        #concat_5 = torch.cat([trans_5, down_1], dim=1) 
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
  in_dim = 11
  out_dim = 6
  
  x = torch.Tensor(1, in_dim, inp_size, inp_size, inp_size)
  x.to(device)
  print("x size: {}".format(x.size()))
  
  model = UNet(in_dim=in_dim, out_dim=out_dim, num_filters=4)
  
  out = model(x)
  print("out size: {}".format(out.size()))
