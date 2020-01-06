
# Grid Free Energy Prediction with ConvNet 

<p align="center">
  <img src="figs/1ycr_orig.gif">
</p>

Prediction of fragment free energy maps from protein structure with ConvNet

## Note to repo admins: :point_up: 

__This repo must be kept clean and organized! Please do not change original functions!__
__If you do change them, keep it for your own use, and don't push it here.  You can__
__only push required tasks, while making sure that your code satisfy the following:__

* Code must be both memory and speed efficient 
* Code must be easily redable, clean
* Every function needs unittest, which validates its correctness!
* All arithmethic operations performed on grid(usually large(100x100x100))
  must be done with torch.cuda()! 

## TODO list:

0. Data Augmentation - Rotations: :fire:
    * 24 90 degree rotation:  __Done!__ :sunglasses: -- *Talant & Arth* 
    * Random ratations - TPL: __Done!__ -- *Talant*  :eyes: :ocean:
    * __We must get reliable, descent results with both!__ 
 
1. Buried vs Surface level hotspot accuracy. __In progress__

2. New methods for frag map baseline correction: __Done!__ *Talant*
    * Median of the edge voxels :thumbsup:
    * Median
    * Mode

3. Test different loss function: __Almost Done__! -- *Sid* 
   * L1
   * Log-Cosh Loss
   * Huber Loss (Smooth L1)
   * Quantile Loss
   * Bin Loss -- *Arth*
   
4. Scatter plots for predicted vs actual frag maps. __Done__! -- *Sid* 

5. Need to prepare a simple tutorial markdown(.md), where we simply
demonstrate what is being done here, including all functionalities,
visualizations etc. Unit tests have perfect examples. The best way
to do this would be using .ipynb and save it as .md file  __In progress__ *All*


6. Prediction of discretized output maps through binning the GFE values. __In Progress__ *Sid & Arth*
   * Loss functions: CrossEntropy or any loss function that handle classification

7. Code maintenance:  __In progress__ *Talant*
   * Rewrite all functions that does numerical arifmetic in torch.cuda()!
   * Make all functions stable! Adjust unit tests!


8. Hyperparam tuning: 
    * kernel size
    * number of conv layers
    * more....


### Main Requirements:
- `python`  3.6 >
- `PyTorch` 
- `cuda-10` 
- `gcc` 7 >  
- `TorchProteinLibrary`
- `PyVista` 
