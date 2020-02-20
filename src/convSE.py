import os
import sys
import torch
from torch import nn
from torch.nn.modules.module import Module
import torch.nn.functional as F
import numpy as np

from se3cnn.image.convolution import SE3Convolution

class FragMapSE3(Module):
    def __init__(self, num_input_channels=11):
        super(FragMapSE3, self).__init__()

        self.conv = nn.Sequential(
            SE3Convolution([(num_input_channels,0)], [(32,0)], size=5, dyn_iso=True, padding=2, bias=None),
            nn.ReLU(),
            # ScalarActivation([(multiplier*2, F.relu)], bias=False),

            SE3Convolution([(32,0)], [(64,0)], size=5, dyn_iso=True, padding=2, bias=None),
            nn.ReLU(),
            # ScalarActivation([(multiplier*2, F.relu)], bias=False),

            SE3Convolution( [(64,0)], [(32,0)], size=5, dyn_iso=True, padding=2, bias=None),
            nn.ReLU(),

            SE3Convolution( [(32,0)], [(6,0)], size=5, dyn_iso=True, padding=2, bias=None),
            nn.ReLU(),

        )


        self.apply(init_weights)


    def forward(self, input):
        conv_out = self.conv(input)
        return conv_out


#custom weigt init
def init_weights(m):
    if type(m) == nn.Conv3d:
        torch.nn.init.xavier_uniform_(m.weight)
    if type(m) == nn.Linear:
        torch.nn.init.xavier_uniform_(m.weight)



def count_parameters(model):
    n_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    return n_params


if __name__=='__main__':

    import pyvista as pv

    ########---------- Simple test with 1 forward pass -----------########
    c, d, h, w = 11, 20, 20, 20
    torch.manual_seed(3000)
    data = torch.randn(1, c, d, h, w)

    print(data.size())

    #invoke the model
    model = ProteinReprSE3()
    #model = CnnModel()
    output = model(data)

    #compare input and output shapes
    print("11 input channels VS 4 output channels")
    print("Input dimension -->", data.size())
    print("Output dimension -->",output.size())


    #plot output density map
    chan_id = 4 # can be 0,1,2,3
    channel = output[0,chan_id,:,:,:].detach().numpy()
    p = pv.Plotter(point_smoothing = True)
    p.add_volume(np.abs(channel), cmap = "viridis", opacity = "linear")

    p.show()

