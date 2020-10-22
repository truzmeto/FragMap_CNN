import torch
import torch.nn as nn
import torch.nn.functional as F

def ConvBlock1(in_dim, out_dim, size, act):
    return nn.Sequential(
        nn.Conv3d(in_dim, out_dim, kernel_size=size, stride=1, padding=size//2 ),
        nn.BatchNorm3d(out_dim),
        act,
    )

def ConvBlock2(in_dim, out_dim, size, act, stride):
    return nn.Sequential(
        ConvBlock1(in_dim, out_dim, size, act),
        nn.Conv3d(out_dim, out_dim, kernel_size=size, stride=stride, padding=size//2), 
        nn.BatchNorm3d(out_dim),
    )


def ConvTransBlock(in_dim, out_dim, size, act):
    return nn.Sequential(
        nn.ConvTranspose3d(in_dim, out_dim, kernel_size=size, stride=2, padding=size//2, output_padding=1),
        nn.BatchNorm3d(out_dim),
        act,
    )



class UNet(nn.Module):
    def __init__(self, in_dim, out_dim, size, mult):
        super(UNet, self).__init__()

        
        m = mult
        act = nn.LeakyReLU(0.2, inplace=True)
        
        # Down sampling
        self.down_1 = ConvBlock2(in_dim, m, size, act, stride=2)
        self.down_2 = ConvBlock2(m, m * 2, size, act, stride=2)
        self.down_3 = ConvBlock2(m * 2, m * 4, size, act, stride=2)
        self.down_4 = ConvBlock2(m * 4, m * 8, size, act, stride=2)
        self.down_5 = ConvBlock2(m * 8, m * 16, size, act, stride=2)

        
        # Bridge
        self.bridge = ConvBlock2(m * 16, m * 32, size, act, stride=1)

        
        # Up sampling
        self.trans_1 = ConvTransBlock(m * 32, m * 32, size, act)
        self.up_1 = ConvBlock2(m * 48, m * 16, size, act, stride=1)
        
        self.trans_2 = ConvTransBlock(m * 16, m * 16, size, act)
        self.up_2 = ConvBlock2(m * 24, m * 8, size, act, stride=1)

        self.trans_3 = ConvTransBlock(m * 8, m * 8, size, act)
        self.up_3 = ConvBlock2(m * 12, m * 4, size, act, stride=1)

        self.trans_4 = ConvTransBlock(m * 4, m * 4, size, act)
        self.up_4 = ConvBlock2(m * 6, m * 2, size, act, stride=1)

        self.trans_5 = ConvTransBlock(m * 2, m * 2, size, act)
        self.up_5 = ConvBlock2(m * 3, m * 1, size, act, stride=1)

        
        # Output
        self.out = ConvBlock1(m, out_dim, size, act)

    
    def skip(self, uped, bypass):
        """
        This functions pads or unpads bypass depending on how different dims are
        between upsampled and bypass.
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
        #print(bridge.shape)
        
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
  inp_size = 96
  in_dim = 11
  out_dim = 6
  
  x = torch.Tensor(1, in_dim, inp_size, inp_size, inp_size)
  x.to(device)
  print("x size: {}".format(x.size()))
  
  model = UNet(in_dim = in_dim, out_dim = out_dim, size = 5, mult = 8)
  
  out = model(x)
  print("out size: {}".format(out.size()))
