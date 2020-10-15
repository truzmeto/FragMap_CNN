import os
import sys
import torch
import numpy as np

from TorchProteinLibrary.FullAtomModel import PDB2CoordsUnordered
from TorchProteinLibrary.FullAtomModel import getBBox

pdb_ids = ["3my5_a", "4f5t", "4wj9", "3bi0", "1d6e_nopep_fixed"]

path = "/u1/home/tr443/data/fragData/"
path_list = [path + i + ".pdb" for i in pdb_ids]


pdb2coords = PDB2CoordsUnordered()
coords, _, resnames, _, atomnames, num_atoms = pdb2coords(path_list)
a, b = getBBox(coords, num_atoms)
c = a-b
c = torch.abs(c).int()
print(c)
