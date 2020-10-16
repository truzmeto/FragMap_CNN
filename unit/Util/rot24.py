import torch
import pyvista as pv
import numpy as np
import sys
import os
from Shapes3D import get3D_rod

sys.path.append(os.path.join(os.path.dirname(__file__), "../.."))
from src.Util.rot24 import Rot90Seq

rod = get3D_rod()

pl = pv.Plotter(point_smoothing = True, shape=(4, 6))
fs = 12
irot = 0

for i in range(4):
    for j in range(6):

        channel = Rot90Seq(rod, iRot=irot)
        ch = channel[0,0,:,:,:].numpy()
        text = 'Rot90, RotIndx = '+str(irot)
        pl.subplot(i, j)
        pl.add_text(text, position = 'upper_left', font_size = fs)
        pl.add_volume(ch, cmap = "viridis_r", opacity = "linear")

        irot = irot + 1

pl.add_axes()
#pl.camera_position = [(0.5,0.5,0.5),
#                      (31.3, 9.8, 20.0),
#                      (0.13, 0.03, 0.99),]
pl.show()
