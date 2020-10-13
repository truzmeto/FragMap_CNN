import torch
import torch.nn as nn
import torch.nn.functional as F

class UNet(nn.Module):
    def encoder_block(self, in_channels, out_channels, kernel_size=3):
        """
        Single 3D encoder(contraction) block
        """

        block = nn.Sequential(
            nn.Conv3d(in_channels = in_channels, out_channels = out_channels, kernel_size = kernel_size),
            nn.ReLU(),
            nn.BatchNorm3d(out_channels),
            nn.Conv3d(in_channels = out_channels, out_channels = out_channels, kernel_size = kernel_size),
            nn.ReLU(),
            nn.BatchNorm3d(out_channels),
        )
        return block


    def decoder_block(self, in_channels, mid_channel, out_channels, kernel_size=3):
        """
        Single decoder(expansion) block
        """
        block = nn.Sequential(
            nn.Conv3d(in_channels = in_channels, out_channels = mid_channel, kernel_size = kernel_size),
            nn.ReLU(),
            nn.BatchNorm3d(mid_channel),
            nn.Conv3d(in_channels = mid_channel, out_channels = mid_channel, kernel_size = kernel_size,),
            nn.ReLU(),
            nn.BatchNorm3d(mid_channel),
            nn.ConvTranspose3d(in_channels = mid_channel, out_channels = out_channels, kernel_size=3, stride=2, padding=1, output_padding=1) #????
        )
        return  block


    def final_block(self, in_channels, mid_channel, out_channels, kernel_size=3):
        """
        Final block
        """

        block = nn.Sequential(
            nn.Conv3d(in_channels = in_channels, out_channels = mid_channel, kernel_size = kernel_size),
            nn.ReLU(),
            nn.BatchNorm3d(mid_channel),
            nn.Conv3d(in_channels = mid_channel, out_channels = mid_channel, kernel_size = kernel_size),
            nn.ReLU(),
            nn.BatchNorm3d(mid_channel),
            nn.Conv3d(in_channels = mid_channel, out_channels = out_channels, kernel_size=kernel_size, padding=1), 
            nn.ReLU(),
            nn.BatchNorm3d(out_channels),
        )
        return  block

    def __init__(self, in_channel, out_channel):
        super(UNet, self).__init__()

        #Encode
        self.conv_encode1 = self.encoder_block(in_channels=in_channel, out_channels=64)
        self.conv_maxpool1 = nn.MaxPool3d(kernel_size=2)
        self.conv_encode2 = self.encoder_block(64, 128)
        self.conv_maxpool2 = nn.MaxPool3d(kernel_size=2)
        self.conv_encode3 = self.encoder_block(128, 256)
        self.conv_maxpool3 = nn.MaxPool3d(kernel_size=2)

        #Bottleneck
        self.bottleneck = nn.Sequential(
            nn.Conv3d(kernel_size=3, in_channels=256, out_channels=512),
            nn.ReLU(),
            nn.BatchNorm3d(512),
            nn.Conv3d(kernel_size=3, in_channels=512, out_channels=512),
            nn.ReLU(),
            nn.BatchNorm3d(512),
            nn.ConvTranspose3d(in_channels=512, out_channels=256, kernel_size=3, stride=2, padding=1, output_padding=1) 
        )

        #Decode
        self.conv_decode3 = self.decoder_block(512, 256, 128)
        self.conv_decode2 = self.decoder_block(256, 128, 64)
        self.final_layer = self.final_block(128, 64, out_channel)

    def crop_and_concat(self, upsampled, bypass, crop=False):
        """

        This layer crop the layer from encoder (contraction) block and concat
        it with decoder(expansion) block vector
        """
        if crop:
            c = (bypass.size()[2] - upsampled.size()[2]) // 2
            bypass = F.pad(bypass, (-c, -c, -c, -c, -c, -c)) #TR: adjusted for 3D
            #bypass = F.pad(bypass, (-c, -c, -c, -c))

        return torch.cat((upsampled, bypass), 1)


    def forward(self, x):
        #Encode
        encode_block1 = self.conv_encode1(x)
        print("encode_b1 shape",encode_block1.shape)
        encode_pool1 = self.conv_maxpool1(encode_block1)
        print("encode_p1 shape",encode_pool1.shape)

        encode_block2 = self.conv_encode2(encode_pool1)
        print("encode_b2 shape",encode_block2.shape)
        encode_pool2 = self.conv_maxpool2(encode_block2)
        print("encode_p2 shape",encode_pool2.shape)

        
        encode_block3 = self.conv_encode3(encode_pool2)
        print("encode_b3 shape",encode_block3.shape)
        encode_pool3 = self.conv_maxpool3(encode_block3)
        print("encode_p3 shape",encode_pool3.shape)
        
        #Bottleneck
        bottleneck1 = self.bottleneck(encode_pool3)
        print("bottleneck shape", bottleneck1.shape)
        
        #Decode
        decode_block3 = self.crop_and_concat(bottleneck1, encode_block3, crop=True)
        print("decode_b3 ", decode_block3.shape)

        cat_layer2 = self.conv_decode3(decode_block3)
        decode_block2 = self.crop_and_concat(cat_layer2, encode_block2, crop=True)
        print("decode_b2 ", decode_block2.shape)
                
        cat_layer1 = self.conv_decode2(decode_block2)
        decode_block1 = self.crop_and_concat(cat_layer1, encode_block1, crop=True)
        print("decode_b1 ", decode_block1.shape)
        
        final_layer = self.final_layer(decode_block1)
        print("final layer ", final_layer.shape)
        
        return  final_layer

if __name__ == "__main__":
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    inp_size = 92
    x = torch.Tensor(1, 3, inp_size, inp_size, inp_size)
    x.to(device)
    print("x size: {}".format(x.size()))
    
    model = UNet(3, 6)
    
    out = model(x)
    print("out size: {}".format(out.size()))
