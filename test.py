import os
import sys
import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import gc

from src.convSE import FragMapSE3
from src.volume import get_volume
from src.mapIO import write_map, greatest_dim
from src.target import get_target
from src.util import grid2vec, unpad_mapc, pad_mapc
from src.loss_fns import PenLoss


def get_inp(pdb_ids, resolution, pdb_path, rotate = False):
    """
    This function takes care of all inputs: 
    volume, maps + rotations etc..

    """
    
    norm = True
    map_path = pdb_path + "maps/"                                                                                
    map_names_list = ["apolar", "hbacc","hbdon", "meoo", "acec", "mamn"]
    dim = greatest_dim(map_path, pdb_ids) + 1
    box_size = int(dim*resolution)
    batch_list = [pdb_path + ids + ".pdb" for ids in pdb_ids]

    with torch.no_grad():

        volume, _ = get_volume(path_list = batch_list, 
                               box_size = box_size,
                               resolution = resolution,
                               norm = norm,
                               rot = False,
                               trans = False)
        
        target, pad, center = get_target(pdb_path,
                                map_names_list,
                                pdb_ids = pdb_ids,
                                maxD = dim)

        
        if rotate:
            irot = torch.randint(0, 24, (1,)).item()
            volume = Rot90Seq(volume, iRot = irot)
            target = Rot90Seq(target, iRot = irot)
                    
    return volume, target, pad, center



def output_maps(output, pad, resolution, center, ipdb, out_path, map_names_list):

    for imap in range(len(map_names_list)):
        
        grid = output[0,imap,:,:,:].cpu().detach().numpy()
        grid = unpad_mapc(grid, pad = pad[0,:])
        
        nx, ny, nz = grid.shape              #get new dims
        vec = grid2vec([nx,ny,nz], grid)     #flatten
        
        #out_name = ipdb + "."+ map_names_list[imap] #+ "P"
        out_name = map_names_list[imap]
        
        write_map(vec, out_path + "maps/", out_name, center[0,:],
                  res = resolution, n = [nx,ny,nz])
        
    return None


def scatter_plot(gfeM, gfeP, pad, pdb_id, map_names):                                      
    '''                                                                                                              
    
    '''                                                                                                              
    
    print('Plotting ', pdb_id, ' scatterplot')                                                                                
    
    fig = plt.figure(figsize = (16,8))                                                                
    plot_fn = pdb_id + '_sp.png'    
    colors = 'bgrcmy'
    plt.title(pdb_id + "  Predicted vs Measured GFE")

    plt.axis('off')  
                                                                                                                  
    
    plt.xlim(-2.2, 2.2)
    plt.ylim(-2.2, 2.2)
    l_x = np.arange(-3.0, 3.0, 0.1)
         
    i = 0
    for imap in map_names:                                                                                               

        i = i + 1
                                                                                     
        gfeP = gfeP[0,imap,:,:,:].cpu().detach().numpy()                                                                        
        x = unpad_mapc(gfeP, pad = pad[0,:])           

        gfeM = gfeM[0,imap,:,:,:].cpu().detach().numpy()
        y = unpad_mapc(gfeM, pad = pad[0,:])           

        #nx, ny, nz = grid.shape              #get new dims                                                                     
        #vec = grid2vec([nx,ny,nz], grid)     #flatten   

        ax = fig.add_subplot(2, 3, i)                                                                                           
        ax.plot(l_x,l_x, color='black')                                                                                         
        plt.scatter(x, y, s = 0.01, color = colors[i-1])

        plt.grid()                                                                                                       
        plt.legend([imap], loc=0)                                                                                   
        plt.ylabel("GFE Measured")                                                                                     
        plt.xlabel("GFE Predicted")                                                                                    
                                                                                                                         
                                                                                                                         
    fig.set_tight_layout(True)                                                                                           
    fig.savefig('output/figs/' + plot_fn)                                                                                       
                                        
    return None


if __name__=='__main__':


    resolution = 1.000
    #RT = 0.59248368    # T=298.15K, R = 0.001987204 kcal / (mol K)
    istate_load = 5000
    dev = 0

    #pdb_path = "/home/tr443/Projects/Frag_Maps/data/"
    pdb_path = "/u1/home/tr443/data/fragData/"

    
    pdb_ids = ["1ycr", "1pw2"]#, "2f6f", "4ic8",
               #"1s4u", "2am9", "3my5_a"]#, "4f5t"]
               #"2zff", "1bvi_m1_fixed"]#,
               #"4djw", "4lnd_ba1", "4obo", "1h1q",
               #"3fly",  "4gih", "2gmx", "4hw3",
               #"3w8m", "2qbs",  "4jv7", "4ypw_prot_nocter",
               #"5q0i", "1r2b", "2jjc",
               #"3bi0", "4wj9", "1d6e_nopep_fixed"]    
    
                   
    map_names_list = ["apolar", "hbacc","hbdon", "meoo", "acec", "mamn"]
    map_path = pdb_path + "maps/"
    out_path = 'output/'
    #map_names_listM = [m + "M" for m in map_names_list] #cleaned maps
    map_names_listP = [m + "P" for m in map_names_list] #predicted maps
    
    params_file_name = str(istate_load) + 'net_params'
    
    torch.cuda.set_device(dev)
    model = FragMapSE3().cuda()
    model.load_state_dict( torch.load( out_path + params_file_name ))#+".pth"))
    model.eval() # Needed to set the model into evaluation mode,
                 # where 1 forward pass is performed using trained weights
    criterion = PenLoss()

    
    for ipdb in pdb_ids:
        
        volume, test_map, pad, center =  get_inp(pdb_ids, resolution, pdb_path, rotate = False)
        output = model(volume)
        #loss = criterion(output, test_map, thresh = 2.0)
        
        #print("Testing Loss", loss.item(), ipdb)
        print("Writing GFE maps for " + ipdb + ".pdb")
        output_maps(output, pad, resolution, center, ipdb, out_path, map_names_listP)
        output_maps(test_map, pad, resolution, center, ipdb, out_path, map_names_list)
        #scatter_plot(test_map, output, pad, pdb_id, map_names_list)

        gc.collect()
        del volume, test_map, output
        torch.cuda.empty_cache() 
