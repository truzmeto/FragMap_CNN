import os
import sys
import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import gc

#import functions from src
from src.cnn  import CnnModel
from src.volume import get_volume
from src.mapIO import greatest_dim
from src.target import get_target
from src.rot24 import Rot90Seq
from src.loss_fns import PenLoss


def get_inp(pdb_ids, pdb_path, irot, train = True):
        
    norm = True
    resolution = 1.000
    #RT = 0.59248368 # T=298.15K, R = 0.001987204 kcal/(mol K)
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
        
        target, _, _ = get_target(map_path,
                                  map_names_list,
                                  pdb_ids = pdb_ids,
                                  maxD = dim)

                
        if train:
            volume = Rot90Seq(volume, iRot = irot)
            target = Rot90Seq(target, iRot = irot)
                    
    return volume, target


def run_model(volume, target, model, train = True):
    
    criterion = PenLoss()
    if train:
        model.train()
        model.zero_grad()
    else:
        model.eval()
        
    output = model(volume)
    loss = criterion(output, target, thresh = 1.0)
            
    if train:
        loss.backward()
        optimizer.step()

    return loss.item()


if __name__=='__main__':

    
    
    lrt = 0.0001
    wd = 0.00001
    max_epoch = 5000
    start = 0
    dev_id = 1    
    nrots = 24

    pdb_path = "../../data/"
    pdb_ids = ["1ycr", "1pw2", "2f6f","4ic8"]
    pdb_val = ["1s4u", "2am9", "3my5_a", "3w8m"]#,"4f5t"]
    
    out_path = 'output/'
    params_file_name = 'net_params'
   
    #invoke model
    torch.cuda.set_device(dev_id)
    model = CnnModel().cuda()
    #model.load_state_dict(torch.load(out_path +str(start)+params_file_name))#+".pth"))
    optimizer = optim.Adam(model.parameters(), lr = lrt, weight_decay = wd)
    
    iStart = start + 1
    iEnd = max_epoch + 1

    
    for epoch in range(iStart, iEnd):

        L = 0.0
        for irot in range(nrots):
            volume, target = get_inp(pdb_ids, pdb_path, irot, train=True)
            lossT = run_model(volume, target, model, train=True)    
            L = L + lossT
        
        if epoch % 10 == 0:
            torch.save(model.state_dict(), out_path + str(epoch) + params_file_name)
            
    
        volumeV, targetV = get_inp(pdb_val, pdb_path, irot=0, train=False)
        lossV = run_model(volumeV, targetV, model, train=False)  
        
        if epoch % 5 == 0:
            with open(out_path+'log_train.txt', 'a') as fout:
                fout.write('%d\t%f\t%f\n'%(epoch, L/(nrots), lossV))
   
        gc.collect()
        del volume, target, volumeV, targetV
        torch.cuda.empty_cache() 
    
       
