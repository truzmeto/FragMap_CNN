
# Grid Free Energy Prediction with ConvNet 

<p align="center">
  <img src="figs/1ycr_orig.gif">
</p>

Prediction of fragment Grid Free Energies (GFE) from protein structure with ConvNets

## Introduction

This repo uses 3D ConvNets as well as their rotationally equivariant SE(3) ConvNets, to learn interactions between
proteins and small molecular fragments. These interactions are represented as free energy maps obtained from the occupancies
of each molecular fragment at each point in space calculated using the SILCS methodology.


## TODO list:

0. SE(3):
    * some net - __running!__
    * U-net
    * ResNet


1. Accuracy estimation: :fire:
    * Show how prediction accuracy changes with $r$!__ :sunglasses:  
    * 1st get bounding box, then calculate box face average difference as size decreases.
    
3. Buried vs Surface level hotspot accuracy. __In progress__
   * Find a way to differentiate surface and buried hot spots!
   

4. Implement U-net and perform training with all available structures: __in progress!__ 
    * Median of the edge voxels :thumbsup:
   

5. Code maintenance:  __In progress__ *Talant*
   * Rewrite all functions that does numerical arifmetic in torch.cuda(). ???

     


### Main Requirements:
- `python`  3.6 >
- `PyTorch` 1.2 >  
- `cuda`  10.0 > 
- `gcc` 7 >  
- `TorchProteinLibrary`
- `PyVista` 
