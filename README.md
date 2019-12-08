# FragMap_CNN
Prediction of fragment free energy maps from protein structure with ConvNet

## TODO list:

1. Write unit test for target.py

2. tensor.transpose().flip() does produce physical volume rotation
without interpolation, but rotations are restricted to be multiple of 90.
Can we generate all 24 ?

3. Check how to invert baseline correction for test map?

4. Add random translations(small values) as part of augmentation. This is easy!

5. Right not testing is done on single protein, might do batches when
data size increase.

6.