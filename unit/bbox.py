import os
import sys
import torch
import numpy as np

from TorchProteinLibrary.FullAtomModel import PDB2CoordsUnordered
from TorchProteinLibrary.FullAtomModel import getBBox


pdb_ids = ["1ycr"]#, "1pw2", "1s4u", "2f6f", "2am9", "3my5_a", "3w8m", "4ic8", "4f5t"]
path = "/u1/home/tr443/data/fragData/" #../../data/"
path_list = [path + i + ".pdb" for i in pdb_ids]




pdb2coords = PDB2CoordsUnordered()
coords, _, resnames, _, atomnames, num_atoms = pdb2coords(path_list)
a, b = getBBox(coords, num_atoms)
c = a-b
c = torch.abs(c).int()#.numpy()
print(c//2)
#print(c[:,0]//2)
#for i in range(len(pdb_ids)):
#    print(a[i].numpy(), b[i].numpy(), c[i].numpy())
    #print(c.max().ceil())
