from util import pad_mapc, vec2grid, grid2vec, pad_mapc, unpad_mapc, load_model
import torch.nn.functional as F
from loss_fns import PenLoss
import torch.optim as optim
from target import get_target
import torch.nn as nn
import torch
from volume import get_volume
from mapIO import read_map, write_map, greatest_dim
import numpy as np
import sys
import os
import pyvista as pv
# sys.path.append(os.path.join(os.path.dirname(__file__), ".."))


# model params
lrt = 0.0001
# lrd = 0.0001
wd = 0.00001

# Cooridinate of the min spot x
# Specifc protein tool 
# Multi hotspot competitive 
# Try a different protein
# 2 hotspot for  different 
# protein - protein interaction using Fragmaps
# Multi layer Grad-Cam - weighted combination of layers
# contrastive hotspots


# relu at intermediate level gives protein structure ???


def get_hotspot(vol):
    # Assumes that user has converted the negative GFE values to positive
    # Returns tensor of location of hotspot


    hotspot_location = (vol == torch.max(vol)).nonzero()
    hotspot_location = hotspot_location.squeeze()

    return hotspot_location

def get_hotspot_list(vol, n_hotspots=5, write_map=False, start_scale=10, step = 5):

    # Create a void volume of shape like input
    hotspot_volume = torch.zeros(size=vol.shape)

    hotspot_locations =[]

    scale = n_hotspots * start_scale
    for i in range(n_hotspots):

        hotspot_location = get_hotspot(vol)
        hotspot_locations.append(hotspot_location)

        hotspot_volume[hotspot_location[0], hotspot_location[1], hotspot_location[2], hotspot_location[3], hotspot_location[4]] += scale

        vol[hotspot_location[0],hotspot_location[1], hotspot_location[2], hotspot_location[3], hotspot_location[4]] = 0

        scale -= step

    return hotspot_volume, hotspot_locations


class CNN_Viz(nn.Module):
    def __init__(self, model, selected_layer=8,  hotspots=True):
        super(CNN_Viz, self).__init__()
        self.hotspots = hotspots
        self.model = model
        self.selected_layer = selected_layer
        self.gradients = None

    def activations_hook(self, outgrads):
        self.gradients = outgrads

    def forward(self, x):

        for index, layer in enumerate(model.conv):
            if index == 0:
                vol = layer(x)
            else:
                vol = layer(vol)
            if index == self.selected_layer:
                h = vol.register_hook(self.activations_hook)

        if self.hotspots:
            vol = F.relu(-vol)
            print(vol.max())
        else:
            vol = F.relu(vol)
                
        return vol

    def get_activations_gradient(self):
        return self.gradients
    
    def get_activations_channels(self, x):
        '''
            Returns activations and # of channels of selected layer
        '''
        channels = 0
        for index, layer in enumerate(self.model.conv):
            if index == 0:
                vol = layer(x)
            else:
                vol = layer(vol)
            if index == self.selected_layer:
                channels = layer.out_channels
                break
        return vol.detach() , channels




def output_maps(output, pad, resolution, ipdb, out_path, map_name_list):

    for imap in range(len(map_names_list)):
        
        grid = output[0,imap,:,:,:].cpu().detach().numpy()
        grid = unpad_mapc(grid, pad = pad[0,:])
        
        nx, ny, nz = grid.shape              #get new dims
        vec = grid2vec([nx,ny,nz], grid)     #flatten
        
        out_name = ipdb + "."+ map_names_list[imap] + "P"
        write_map(vec, out_path + "maps/", out_name, center[0,:],
                  res = resolution, n = [nx,ny,nz])
        
    return None

if __name__ == '__main__':

    resolution = 1.000
    kBT = 0.592  #T=298K, kB = 0.001987 kcal/(mol K)

    pdb_path = '../data/'
    pdb_ids = ["1ycr",]#"1pw2", "2f6f", "4f5t", "1s4u", "2am9", "3my5_a", "3w8m", "4ic8"]

    map_names_list = ["apolar", "hbacc","hbdon", "meoo", "acec", "mamn"]
    map_path = '../data/maps/' 
    out_path = '../output/'

    params_file_name = 'net_params_2k_PenLoss_PReLU_k7_2s'
    test_indx = 0

    dim = greatest_dim(map_path, pdb_ids) + 1
    box_size = int(dim*resolution)
    batch_list = [pdb_path+pdb_ids[test_indx]+".pdb"] 

    # get volume tensor
    norm = True
    volume , _ = get_volume(path_list = batch_list, 
                        box_size = box_size,
                        resolution = resolution,
                        norm = norm,
                        rot = False,
                        trans = False)

    # get testing map tensor
    map_norm = False
    test_map, pad, gfe_min, gfe_max, center = get_target(map_path,
                                            map_names_list,
                                            pdb_ids = [pdb_ids[test_indx]],
                                            maxD = dim,
                                            kBT = kBT,
                                            density = False,
                                            map_norm = map_norm)

    # #convert target maps to torch.cuda
    test_map = torch.from_numpy(test_map).float().cuda()
    # print((test_map == torch.min(test_map)).nonzero())

    selected_layer = 0

    torch.cuda.set_device(0)
    model = load_model(out_path, params_file_name)
    model.eval() #Needed to set into inference mode

    print("Model:\n", model)
    conv_output = torch.zeros(volume.size())
    # Fully connected layer is not needed
    optimizer = optim.Adam(model.parameters(), lr = lrt, weight_decay = wd )
    criterion = PenLoss() #nn.L1Loss()


    cnn_viz = CNN_Viz(model, selected_layer= selected_layer ,hotspots=True)

    pred = cnn_viz(volume)

    # Hotspot location information
    # hotspot_volume, hotspot_location = get_hotspot(pred)

    # pred[hotspot_location[0],hotspot_location[1], hotspot_location[2], hotspot_location[3], hotspot_location[4]] = 0

    # hotspot_volume, hotspot_location = get_hotspot(pred)

    n_hotspots = 15
    hotspot_volume, hotspot_locations = get_hotspot_list(pred, n_hotspots=n_hotspots)

    for i in range(n_hotspots):
        print("Hotspot location: ",hotspot_locations[i])



    max_t = torch.max(pred)
    max_t.backward()


    # pull the gradients out of the model
    gradients = cnn_viz.get_activations_gradient()
    print("Gradients Shape: ", gradients.shape)

    # pool the gradients across the channels
    pooled_gradients = torch.mean(gradients, dim=[0, 2,3,4])

    activations, out_channels = cnn_viz.get_activations_channels(volume)

    print("Activations Shape: ",activations.shape)

    # weight the channels by corresponding gradients
    for i in range(out_channels):
        activations[:, i, :, :] *= pooled_gradients[i]


     # average the channels of the activations
    heatmap = torch.mean(activations, dim=1).squeeze()

    # # relu on top of the heatmap
    # # expression (2) in https://arxiv.org/pdf/1610.02391.pdf
    heatmap = F.relu(heatmap)

    heatmap = heatmap / torch.max(heatmap)


    # Heatmap to GFE file
    grid = heatmap[:,:,:].cpu().detach().numpy()
    grid = unpad_mapc(grid, pad = pad[0,:])

    nx, ny, nz = grid.shape              #get new dims
    vec = grid2vec([nx,ny,nz], grid)     #flatten

    out_name = "1ycr" + "."+ str(selected_layer) + ".heatmap"
    # write_map(vec, out_path + "../data/maps/", out_name, center[0,:],
    #       res = resolution, n = [nx,ny,nz])

    # Hotspot to GFE file
    grid = hotspot_volume[0,hotspot_locations[0][1],:,:,:].cpu().detach().numpy()
    grid = unpad_mapc(grid, pad = pad[0,:])

    nx, ny, nz = grid.shape              #get new dims
    vec = grid2vec([nx,ny,nz], grid)     #flatten

    out_name = "1ycr" + "."+ str(selected_layer) + "."+str(n_hotspots)+".hotspots"
    write_map(vec, out_path + "../data/maps/", out_name, center[0,:],
          res = resolution, n = [nx,ny,nz])


    volume_sum= np.sum(volume.cpu().numpy(), axis=1)[0]


    p = pv.Plotter(point_smoothing = True, shape=(2, 1))
    p.subplot(0,0)
    # p.add_volume(hotspot_volume.detach().cpu().numpy()[0,,:,:,:])
    p.add_volume(heatmap.detach().cpu().numpy(), cmap = "hot", opacity = "linear")
    p.add_text("Grad-CAM"+ " layer:"+str(selected_layer)+ \
        " filters:"+str(out_channels), position = 'upper_left', font_size = 16)
    p.subplot(1,0)
    p.add_volume(volume_sum, cmap = "viridis", opacity = "linear")
    p.add_text("Summed-Volume", position = 'upper_left', font_size = 16)
    p.link_views()
    # p.show()



# -------------------------------------------------------------------------
# Filter vis
# -------------------------------------------------------------------------
 
    # selected_layer = 8
    # filter_pos = 2
    # conv_output = torch.zeros(volume.size())
    # # Fully connected layer is not needed
    # pretrained_model = model
    # optimizer = optim.Adam(model.parameters(), lr = lrt, weight_decay = wd )
    # criterion = PenLoss() #nn.L1Loss()


    # x = volume
    # for index, layer in enumerate(model.conv):
    #     print(index)
    #     if index == 0 :
    #         vol_np = layer(volume)
    #     else:
    #         vol_np = layer(vol_np)
    #     if index == selected_layer:
    #         print(vol_np.shape)


    #         vol_numpy = vol_np.detach().cpu().numpy()
    #         p = pv.Plotter(point_smoothing = True, shape=(1, 1))
    #         p.subplot(0,0)
    #         p.add_volume(vol_numpy[0,filter_pos,25:45,:,:], cmap = "viridis", opacity = "linear")
    #         p.add_text(str(layer)+" "+str(filter_pos), position = 'upper_left', font_size = 16)
    #         p.show()

    #     if index == 8:
    #         optimizer.zero_grad()
    #         output = model(volume)
    #         loss = criterion(output, test_map, 1.0)
    #         # loss_values.append(loss.item())
    #         loss.backward()
    #         optimizer.step()
