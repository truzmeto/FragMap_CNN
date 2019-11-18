import os
import sys
import torch
import torch.nn as nn
from src.cnn  import CnnModel
from src.volume import get_volume
from src.mapIO import get_target, write_map
from src.util import grid2vec, unpad_map
import torch.optim as optim
import pyvista as pv


lrt = 0.001
#lrd = 0.0001
#wd = 0.0
max_epoch = 100

torch.cuda.set_device(0)

#get input
pdb_path = "data/"
pdb_id = "1ycr"
path1 = pdb_path+pdb_id+".pdb"
pdb_path_list = [path1]
box_size = 57  # prog complains if box_size is float !!!!!!!!! 
resolution = 1.000
data = get_volume(pdb_path_list, box_size, resolution)
print(data.size())

#get target
map_path = "data/maps/"
map_names_list = ["benc","prpc", "hbacc", "hbdon"]
dim = int(box_size/resolution)

#get padded target fragmap volumes
target, pad, dmin, dsize = get_target(map_path,
                            map_names_list,
                            pdb_id,
                            batch = 1,
                            dim = dim,
                            cutoff=False)
print(target.shape)

target = torch.from_numpy(target).float().cuda()

#invoke model
model = CnnModel().cuda()
criterion = nn.MSELoss()
#criterion = nn.L1Loss()
#optimizer = optim.Adam(model.parameters(), lr=lrt)
optimizer = optim.SGD(model.parameters(), lr=lrt, momentum=0.9)


for epoch in range(max_epoch):

    optimizer.zero_grad()
    output = model(data)
    loss = criterion(output, target)
    loss.backward()
    optimizer.step()

    if epoch % 20 == 0:
        print('Epoch {}, loss {}'.format(epoch, loss.item()))
	

#save trained parameters        
save_path = './output/map_net.pth'
torch.save(model.state_dict(), save_path)

        
#save density maps to file
out_path = "output/"
ori = [40.250, -8.472, 20.406]
res = resolution

for i in range(len(map_names_list)):

    out_name = pdb_id+"."+ map_names_list[i]
    grid = output[0,i,:,:,:].cpu().detach().numpy()
    #grid = unpad_map(grid, xpad=pad[0], ypad=pad[1], zpad=pad[2])

    #un-normalize predicted densities
    vol = grid*dsize[i] + dmin[i]
    
    nx, ny, nz = grid.shape
 
    vec = grid2vec([nx,ny,nz], vol)
    write_map(vec, out_path, out_name, ori, res, n = [nx,ny,nz])
    