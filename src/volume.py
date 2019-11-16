import os
import sys
import torch
from TorchProteinLibrary.Volume import TypedCoords2Volume
from TorchProteinLibrary.FullAtomModel import PDB2CoordsUnordered, Coords2TypedCoords
from TorchProteinLibrary.FullAtomModel import getRandomRotation, getRandomTranslation
from TorchProteinLibrary.FullAtomModel import CoordsRotate, CoordsTranslate, getBBox

def get_volume(path_list, box_size, resolution):
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

    return volume


if __name__=='__main__':

    import pyvista as pv
    
    #A simple unit test!
    path_list = ["../data/1ycr.pdb"]
    box_size = 58  # prog complains if box_size is float !!!!!! 
    resolution = 1.0
    volume = get_volume(path_list, box_size, resolution)
    print(volume.shape)

    Agroup_names = ["Sulfur/Selenium", "Nitrogen Amide",
                    "Nitrogen Aromatic", "Nitrogen Guanidinium",
                    "Nitrogen Ammonium", "Oxygen Carbonyl",
                    "Oxygen Hydroxyl", "Oxygen Carboxyl",
                    "Carbon sp2","Carbon Aromatic",
                    "Carbon sp3"]
    
    chan_id = 10 # Atomic group ids, range 0-10
    channel = volume[0,chan_id,:,:,:].cpu().numpy()
    p = pv.Plotter(point_smoothing = True)
    text = Agroup_names[chan_id]
    p.add_text(text, position='upper_left', font_size=18)
    p.add_volume(channel, cmap = "viridis", opacity = "linear")
    p.show()

