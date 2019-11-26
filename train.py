import os
import sys
import torch
import torch.nn as nn
from src.cnn  import CnnModel
from src.volume import get_volume
from src.mapIO import get_target, write_map
from src.util import grid2vec, unpad_map
from unit.viz import Visualizations
import torch.optim as optim
import pyvista as pv
import numpy as np

torch.manual_seed(11)

lrt_adadelta = 1
wd_adadelta = 0
lrt = 0.001
wd = 0.00001
max_epoch = 5000

torch.cuda.set_device(0)

#get input
pdb_path = "data/"
pdb_id = "1ycr"
path1 = pdb_path+pdb_id+".pdb"
pdb_path_list = [path1]
box_size = 57  # prog complains if box_size is float !!!!!!!!! 
resolution = 1.000
data = get_volume(pdb_path_list, box_size, resolution)
#print(data.size())

#get target
map_path = "data/maps/"
map_names_list = ["benc","prpc", "hbacc", "hbdon"]
dim = int(box_size/resolution)

#get padded target fragmap volumes
target, pad = get_target(map_path,
                         map_names_list,
                         pdb_id,
                         batch = 1,
                         dim = dim,
                         cutoff = True)


target = torch.from_numpy(target).float().cuda()

#invoke model
model = CnnModel().cuda()
criterion = nn.MSELoss()
#criterion = nn.L1Loss()
# optimizer = optim.Adam(model.parameters(), lr=lrt, weight_decay = wd )
optimizer = optim.Adadelta(model.parameters(), lr=lrt_adadelta, weight_decay = wd_adadelta, rho=0.9, eps=1e-06)
vis = Visualizations()
loss_values = []

for epoch in range(max_epoch):

    optimizer.zero_grad()
    output = model(data)
    loss = criterion(output, target)
    loss.backward()
    optimizer.step()
    vis.plot_loss(loss.item(), epoch)

    if epoch % 20 == 0:
        print('Epoch {}, loss {}'.format(epoch, loss.item()))
        

#save trained parameters        
save_path = './output/map_net.pth'
torch.save(model.state_dict(), save_path)

        
#save density maps to file
out_path = "data/maps/"
ori = [40.250, -8.472, 20.406]
res = resolution

for i in range(len(map_names_list)):

    out_name = pdb_id+"."+ map_names_list[i]+"_o"
    grid = output[0,i,:,:,:].cpu().detach().numpy()
    grid = unpad_map(grid, xpad = pad[0], ypad = pad[1], zpad = pad[2])

    #convert from Free-E to density 
    grid[grid <= 0.000] = 0.0001
    kBT = 0.6
    vol = -kBT *np.log(grid)  
    
    nx, ny, nz = grid.shape
 
    vec = grid2vec([nx,ny,nz], vol)
    write_map(vec, out_path, out_name, ori, res, n = [nx,ny,nz])
    
