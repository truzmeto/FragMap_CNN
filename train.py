import os
import sys
import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np

#import functions from src
from src.volume import get_volume
from src.mapIO import greatest_dim
from src.target import get_target
from src.rot24 import Rot90Seq
from src.loss_fns import PenLoss
from src.convSE import FragMapSE3


def get_inp(pdb_ids, pdb_path, rotate = True):
    """
    This function takes care of all inputs: 
    volume, maps + rotations etc..

    """
    
    norm = True
    resolution = 1.000
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
        
        target, _, _ = get_target(pdb_path,
                                  map_names_list,
                                  pdb_ids = pdb_ids,
                                  maxD = dim)

        
        if rotate:
            irot = torch.randint(0, 24, (1,)).item()
            volume = Rot90Seq(volume, iRot = irot)
            target = Rot90Seq(target, iRot = irot)
                    
    return volume, target


def run_model(volume, target, model, criterion, train = True):
    
    if train:
        model.train()
        model.zero_grad()
    else:
        model.eval()

        
    output = model(volume)
    L1 = criterion(output, target, thresh = 1.0)

    
    L1_reg = 0.0
    for w in model.parameters():
        L1_reg = L1_reg + w.norm(1)
   
    Lambda = 0.000001
    loss = L1 + Lambda * L1_reg
   
    if train:
        loss.backward()
        optimizer.step()
    
    return L1.item()


if __name__=='__main__':
    
    
    lrt = 0.0001
    #wd = 0.00001
    max_epoch = 5000
    start = 0
    dev_id = 1    

    pdb_path = "../../data/"
    pdb_ids = ["1ycr", "1pw2", "2f6f", "4ic8", "1s4u", "2am9", "3w8m"]
    pdb_val = ["3my5_a", "4f5t"]
    
    out_path = 'output/'
    params_file_name = 'net_params'
   
    torch.cuda.set_device(dev_id)
    model = FragMapSE3().cuda()
    criterion = PenLoss()

    #uncomment line below if need to load saved parameters
    #model.load_state_dict(torch.load(out_path +str(start)+params_file_name))#+".pth"))

    
    optimizer = optim.Adam(model.parameters(), lr = lrt)#, weight_decay = wd)
    iStart = start + 1
    iEnd = max_epoch + 1
    
    for epoch in range(iStart, iEnd):
            
        volume, target = get_inp(pdb_ids, pdb_path, rotate = False)
        lossT = run_model(volume, target, model, criterion, train = True)    
                          
        volumeV, targetV = get_inp(pdb_val, pdb_path, rotate = True)
        lossV = run_model(volumeV, targetV, model, criterion, train = False)  
       
        if epoch % 20 == 0:
            torch.save(model.state_dict(), out_path + str(epoch) + params_file_name)
         
        if epoch % 2 == 0:
            with open(out_path + 'log_train_val.txt', 'a') as fout:
                fout.write('%d\t%f\t%f\n'%(epoch, lossT, lossV))

    
       
