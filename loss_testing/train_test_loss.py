import os
import sys
import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np

from train import *
from test import *

sys.path.insert(0, '..')
from src.cnn  import CnnModel, CnnModel_Leaky
from src.volume import get_volume
from src.mapIO import greatest_dim, write_map
from src.target import get_target
from src.util import grid2vec, sample_batch, unpad_mapc
from src.loss_fns import *
from visual.scatter import *





################################################
#model params
lrt = 0.0001
wd = 0.00001
max_epoch = 1000
batch_size = 1 #number of structures in a batch

norm = True
map_norm = False 
rand_rotations = False

nsample = 1

################################################
#physical params
resolution = 1.000
T = 298.15 ##in Kelvin
R = 1.98720425864083*10**-3
RT = R*T

################################################

print(os.getcwd())
#paths
pdb_path = '../data/'
pdb_ids = ["1ycr"] #, "1pw2", "2f6f", "4f5t", "1s4u", "2am9", "3my5_a", "3w8m"]#,"4ic8"]

map_names_list = ["apolar", "hbacc","hbdon", "meoo", "acec", "mamn"]
map_path = '../data/maps/' 

out_path = '../output/'

dim = greatest_dim(map_path, pdb_ids) + 1
box_size = int(dim*resolution)

test_indx = 0
batch_list = [pdb_path+pdb_ids[test_indx]+".pdb"] 

################################################

#invoke model
torch.cuda.set_device(0)
model = CnnModel_Leaky().cuda()
optimizer = optim.Adam(model.parameters(), lr = lrt, weight_decay = wd )


################################################
#Scatter plot parameters

frag_names = ["Gen. Apolar","Gen. Acceptor",
                "Gen. Donor","Methanol Oxy",
                "Acec", "Methylammonium Nitro"]

frag_names_short = ["apolar", "hbacc", "hbdon", "meoo","acec", "mamn"]


################################################

if __name__ == '__main__':
    
    loss_list = [
             #nn.MSELoss(),
             #nn.L1Loss(),
             #nn.SmoothL1Loss(),
             #logCoshLoss(),
             #XTanhLoss(),
             XSigmoidLoss(),
             
             ]
    param_name_list = [ 
            #'1500net_params_4k_15_xnorm_xlast_rotB_params.pth', # arths model1
            #'2500net_params_4k_15_xnorm_xlast_params.pth', # arths model2
            
            #'MSE_params.pth',
            #'L1_params.pth',
            #'SmoothL1_params.pth',
            #'logCosh_params.pth',
            #'XTanhLoss_params.pth',
            'XSigmoidLoss_params.pth',
            
            ]

    for criterion in range(len(loss_list)):
        
        param_prefix = param_name_list[criterion][:-11]
        
        pmap_dir = out_path+param_prefix+'_loss_maps/'

        try:
            os.mkdir(pmap_dir)
            print("Directory " , pmap_dir ,  " Created ")
        except FileExistsError:
            print("Directory " , pmap_dir ,  " already exists") 
                
        
        train_model(max_epoch, nsample, batch_size, pdb_ids, pdb_path, 
                    box_size, resolution, norm, rand_rotations, 
                    map_path, map_names_list, dim, RT, map_norm,
                    optimizer, model, loss_list[criterion],
                    out_path, param_name_list[criterion])
        
        test_model(pdb_path, pdb_ids, map_names_list, map_path, norm,
                    batch_list, box_size, resolution,
                    test_indx, dim, RT, map_norm,
                    out_path, param_name_list[criterion], loss_list[criterion], pmap_dir)
        
        scatter_plot(pdb_ids[test_indx], frag_names, frag_names_short, map_path, pmap_dir, param_prefix)
