import os
import sys
import torch
from TorchProteinLibrary.Volume import TypedCoords2Volume, VolumeRotation
from TorchProteinLibrary.FullAtomModel import PDB2CoordsUnordered, Coords2TypedCoords
from TorchProteinLibrary.FullAtomModel import getRandomRotation, getRandomTranslation
from TorchProteinLibrary.FullAtomModel import CoordsRotate, CoordsTranslate, getBBox


def get_volume(path_list, box_size, resolution, norm = True, rot = False, trans = False):
    """
    This function invokes modules from TorchPotentialLibrary and
    reads .pdb inputs, projects atomic coordinates into
    11 density maps in a 3D grid of given size(box_size) with
    given resolution.
    11 ---> 11 atomic groups
   
    output: torch tensor(batch_size, 11, box_dim, box_dim, box_dim)

    """
    
    pdb2coords = PDB2CoordsUnordered()
    assignTypes = Coords2TypedCoords()
    translate = CoordsTranslate()
    rotate = CoordsRotate()
    project = TypedCoords2Volume(box_size, resolution)

    batch_size = len(path_list)
    coords, _, resnames, _, atomnames, num_atoms = pdb2coords(path_list)
    
    a,b = getBBox(coords, num_atoms)
    protein_center = (a+b)*0.5
    coords = translate(coords, -protein_center, num_atoms)

    #rotate xyz 
    if rot:
        random_rotations = getRandomRotation(batch_size)
        coords = rotate(coords, random_rotations, num_atoms)


    box_center = torch.zeros(batch_size, 3, dtype=torch.double, device='cpu').fill_(resolution*box_size/2.0)
    coords = translate(coords, box_center, num_atoms)
    

    #translate xyz
    if trans:                                                                                                      
        random_translations = getRandomTranslation(a, b, resolution*box_size)                                        
        coords = translate(coords, random_translations, num_atoms)                                                   


    coords, num_atoms_of_type, offsets = assignTypes(coords.to(dtype=torch.float32),
                                                     resnames, atomnames, num_atoms)

    volume = project(coords.cuda(), num_atoms_of_type.cuda(), offsets.cuda())
   
    #if rot: #apply rand rots to grid
    #    volume_rotate = VolumeRotation(mode = 'bilinear')
    #    R = getRandomRotation(batch_size)
    #    volume = volume_rotate(volume, R.to(dtype = torch.float, device = 'cuda'))
    #    
        
            
    if norm: #apply min-max norm 
        volume = (volume - torch.min(volume)) / (torch.max(volume) - torch.min(volume))
        

    #TODO: must return rotation and translation matrices!

    return volume
    
    
if __name__=='__main__':
    print("Skibidi Wa Pa Pa!!!")
