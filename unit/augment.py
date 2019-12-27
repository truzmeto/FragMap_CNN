import os
import sys
import torch
import pyvista as pv
import numpy as np
from rot import get3D_rod

sys.path.append(os.path.join(os.path.dirname(__file__), ".."))
from src.augment import get_random_rotation
rod = get3D_rod()


################## compare rotated with original
#p = pv.Plotter(point_smoothing = True, shape=(1, 2))
#fs = 12
#channel = rod[0,0,:,:,:].numpy()
#text = 'original'
#p.subplot(0, 0)
#p.add_text(text, position = 'upper_left', font_size = fs)
#p.add_volume(channel, cmap = "viridis_r", opacity = "linear")
#
#rot_rod , _ = get_random_rotation(rod, rod)
#channel = rot_rod[0,0,:,:,:].numpy()
#text = 'rotated'
#p.subplot(0, 1)
#p.add_text(text, position = 'upper_left', font_size = fs)
#p.add_volume(channel, cmap = "viridis_r", opacity = "linear")
#p.add_axes()
#p.show()


############# plot rotated 24 times ###############################
pl = pv.Plotter(point_smoothing = True, shape=(4, 6))
fs = 12

for i in range(4):
    for j in range(6):

        channel , _ = get_random_rotation(rod, rod)
        ch = channel[0,0,:,:,:].numpy()
        text = 'Rot90 Arth' 
        pl.subplot(i, j)
        pl.add_text(text, position = 'upper_left', font_size = fs)
        pl.add_volume(ch, cmap = "viridis_r", opacity = "linear")

pl.add_axes()
pl.show()
