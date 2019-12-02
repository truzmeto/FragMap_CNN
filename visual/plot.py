import numpy as np
import matplotlib.pyplot as plt
import matplotlib
import os
import sys
import pandas as pd

##=====================================================================
plot_fn= 'plot.pdf'
fig = plt.figure(figsize=(5.,3.5))

dat = pd.read_csv("../output/loss.txt", sep=",", header=None)
dat.columns=["Epoch","Loss"]
x = dat["Epoch"].values
y = dat["Loss"].values

plt.xlabel('Epoch', labelpad=2)
plt.ylabel('Loss', labelpad=2)
plt.plot(x, y, ls='--')


fig.set_tight_layout(True)
plt.show()
fig.savefig(plot_fn)
os.system("epscrop %s %s" % (plot_fn, plot_fn))
