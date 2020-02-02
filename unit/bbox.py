import os
import sys
import torch
import numpy as np

from TorchProteinLibrary.FullAtomModel import PDB2CoordsUnordered
from TorchProteinLibrary.FullAtomModel import getBBox


pdb_ids = ["1ycr", "1pw2", "1s4u", "2f6f", "2am9", "3my5_a", "3w8m", "4ic8"]#, "4f5t"]
path = "/u1/home/tr443/data/fragData/" #../../data/"
path_list = [path + i + ".pdb" for i in pdb_ids]




pdb2coords = PDB2CoordsUnordered()
coords, _, resnames, _, atomnames, num_atoms = pdb2coords(path_list)
a,b = getBBox(coords, num_atoms)

c = b-a
#print(c)
print(c.max().ceil())
