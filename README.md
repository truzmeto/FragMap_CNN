# FragMap_CNN
Prediction of fragment free energy maps from protein structure with ConvNet

## TODO list:

0. Debugging!

1. Need to prepare a simple tutorial markdown(.md), where we simply
demonstrate what is being done here, including all functionalities,
visualizations etc. Unit tests have perfect examples. The best way
to do this would be using .ipynb and save it as .md file

2. tensor.transpose().flip() does produce physical volume rotation
without interpolation, but rotations are restricted to be multiple of 90.
Can we generate all 24 ?

3. Check how to invert baseline correction for test map?

4. Add random translations(small values) as part of augmentation. This is easy!

5. Right now testing is done on single protein, might do in batches when
data size increase. Generlize!

6. Hyperparam tuning.