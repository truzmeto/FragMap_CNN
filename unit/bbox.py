import os
import sys
import torch
import numpy as np

from TorchProteinLibrary.FullAtomModel import PDB2CoordsUnordered
from TorchProteinLibrary.FullAtomModel import getBBox

pdb_ids = ["3my5_a", "4f5t", "4wj9", "3bi0", "1d6e_nopep_fixed",
           "1ycr", "1pw2", "2f6f", "4ic8", "1s4u", "2am9",
           "1bvi_m1_fixed", "4djw", "4lnd_ba1", "4obo",
           "1h1q", "3fly", "4gih", "2gmx", "4hw3", "4ypw_prot_nocter",
            "3w8m", "2qbs", "4jv7", "5q0i", "1r2b", "2jjc"]

path = "/u1/home/tr443/data/fragData/"
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
