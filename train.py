import os
import sys
import torch
import torch.nn as nn
from src.cnn  import CnnModel
from src.volume import get_volume
from src.mapIO import get_target, write_map
from src.util import grid2vec, unpad_map
import torch.optim as optim
import numpy as np
from TorchProteinLibrary.Volume import VolumeRotation
from TorchProteinLibrary.FullAtomModel import getRandomRotation


lrt = 0.0001
#lrd = 0.0001
wd = 0.00001
max_epoch = 5000

torch.cuda.set_device(0)

#get input data
#pdb_path = 'data/'
pdb_path = "/scratch/tr443/fragmap/data/"                                                          

pdb_id = "1ycr"
path1 = pdb_path+pdb_id+".pdb"
pdb_path_list = [path1]
box_size = 57  # prog complains if box_size is float !!!!!!!!! 
resolution = 1.000
data = get_volume(pdb_path_list, box_size, resolution)
#normalize togather
volume = (data - torch.min(data)) / (torch.max(data) -  torch.min(data))


#get target
#map_path = 'data/maps/' 
map_path = "/scratch/tr443/fragmap/data/maps/"                                               

map_names_list = ["apolar", "hbacc", "hbdon", "meoo","acec", "mamn"]
dim = int(box_size/resolution)

#get padded target fragmap volumes
target, pad, gfe_min, gfe_max = get_target(map_path,
                                map_names_list,
                                pdb_id,
                                batch = 1,
                                dim = dim,
                                cutoff = False,
                                density = False)

target = torch.from_numpy(target).float().cuda()

#invoke model
model = CnnModel().cuda()
criterion = nn.MSELoss()
#criterion = nn.L1Loss()
optimizer = optim.Adam(model.parameters(), lr=lrt, weight_decay = wd )
#optimizer = optim.SGD(model.parameters(), lr = lrt, momentum = 0.9)


volume_rotate = VolumeRotation(mode='bilinear')
nrot = 10
for irot in range(nrot):
    #apply random rotations to input                                                          
    R = getRandomRotation(len(pdb_path_list)) #per batch                                      
    data = volume_rotate(volume, R.to(dtype=torch.float, device='cuda'))
    
    for epoch in range(max_epoch):
        optimizer.zero_grad()
        output = model(data)
        loss = criterion(output, target)
        loss.backward()
        optimizer.step()
    
        if epoch % 40 == 0:
            print('{0}, {1}'.format(epoch, loss.item()))
            
#save trained parameters        
save_path = '/scratch/tr443/fragmap/output/map_net.pth'

torch.save(model.state_dict(), save_path)

        
#save density maps to file
#out_path = 'output/'
out_path = "/scratch/tr443/fragmap/output/"  

ori = [40.250, -8.472, 20.406] ###### ???????????
res = resolution
kBT = 0.592 # T=298K, kB = 0.001987 kcal/(mol K)

for i in range(len(map_names_list)):

    out_name = pdb_id+"."+ map_names_list[i]
    grid = output[0,i,:,:,:].cpu().detach().numpy()
    grid = unpad_map(grid, xpad = pad[0], ypad = pad[1], zpad = pad[2])

    #convert from Free-E to density 
    #grid[grid <= 0.000] = 0.0001
    #vol = grid #-kBT *np.log(grid)  
    vol = grid*(gfe_max[i] - gfe_min[i]) + gfe_min[i] 
    #vol = -vol
    nx, ny, nz = grid.shape
 
    vec = grid2vec([nx,ny,nz], vol)
    write_map(vec, out_path, out_name, ori, res, n = [nx,ny,nz])
    
