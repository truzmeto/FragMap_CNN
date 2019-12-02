import os
import sys
import torch
from TorchProteinLibrary.Volume import TypedCoords2Volume
from TorchProteinLibrary.FullAtomModel import PDB2CoordsUnordered, Coords2TypedCoords
from TorchProteinLibrary.FullAtomModel import getRandomRotation, getRandomTranslation
from TorchProteinLibrary.FullAtomModel import CoordsRotate, CoordsTranslate, getBBox

def get_volume(path_list, box_size, resolution, norm = True):
    """
    This function invokes modules from TorchPotentialLibrary and
    reads .pdb inputs, projects atomic coordinates into
    11 density maps in a 3D grid of given size(box_size) with
    given resolution.
    11  --> 11 atomic groups
    
    output: torch tensor(1, 11, box_dim, box_dim, box_dim)

    """
    
    pdb2coords = PDB2CoordsUnordered()
    assignTypes = Coords2TypedCoords()
    translate = CoordsTranslate()
    rotate = CoordsRotate()
    project = TypedCoords2Volume(box_size, resolution)
    
    data_path = path_list    
    coords, _, resnames, _, atomnames, num_atoms = pdb2coords(data_path)
    
    a,b = getBBox(coords, num_atoms)
    protein_center = (a+b)*0.5
    coords = translate(coords, -protein_center, num_atoms)

    pos = box_size*resolution // 2
    new_center = torch.tensor([[pos, pos, pos]], dtype=torch.double)
    coords_ce = translate(coords, new_center, num_atoms)


    coords, num_atoms_of_type, offsets = assignTypes(coords_ce.to(dtype=torch.float32),
                                                     resnames,
                                                     atomnames,
                                                     num_atoms)
    
    volume = project(coords.cuda(),
                     num_atoms_of_type.cuda(),
                     offsets.cuda())

    if norm == True: #apply min-max norm 
        volume = (volume - torch.min(volume)) / (torch.max(volume) -  torch.min(volume))
        
    return volume


if __name__=='__main__':
    print("Code daily!")
