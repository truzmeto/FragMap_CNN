import os
import sys
import torch
import torch.nn as nn

sys.path.append(os.path.join(os.path.dirname(__file__), ".."))

from src.cnn  import CnnModel
from src.volume import get_volume
from src.mapIO import get_target
import torch.optim as optim
import pyvista as pv


lrt = 0.001
lrd = 0.0001
wd = 0.0
max_epoch = 21

torch.cuda.set_device(0)

#get input
pdb_path = "data/"
pdb_id = "1ycr"
path1 = pdb_path+pdb_id+".pdb"
pdb_path_list = [path1]
box_size = 57  # prog complains if box_size is float !!!!!!!!! 
resolution = 1.0
dim=box_size
data = get_volume(pdb_path_list, box_size, resolution)
print(data.shape)


target = 0.1*torch.randn(1, 4, dim, dim, dim).cuda()#+data[:,0:4,:,:,:] #the 0:4 atomic groups
#target = torch.from_numpy(target).float().cuda()
print(target.shape)


model = CnnModel().cuda()
criterion = nn.L1Loss(reduction='sum')
optimizer = optim.Adam(model.parameters(), lr=lrt)

for epoch in range(max_epoch):
    optimizer.zero_grad()
    
    output = model(data)
    loss = criterion(output, target)
    loss.backward()
    optimizer.step()

    if epoch%10 == 0:
        print('Epoch {}, loss {}'.format(epoch, loss.item()))
	

#plot output density map
print(output.shape)
print(output[0,1,:,:,:].cpu().detach().numpy().sum())
#chan_id = 1 # can be 0,1,2,3
#channel = output[0,chan_id,:,:,:].cpu().detach().numpy()
#p = pv.Plotter(point_smoothing=True)
#p.add_volume(channel, cmap="viridis", opacity="linear")
#p.show()

    
		
