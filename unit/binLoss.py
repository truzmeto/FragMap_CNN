import numpy as np
import sys
import os
import pyvista as pv
import matplotlib.pyplot as plt
import seaborn as sns

sys.path.append(os.path.join(os.path.dirname(__file__), ".."))
from src.loss_fns import  get_bin_frequency
from src.mapIO import read_map


sns.set(style="white", palette="muted", color_codes=True)
pdb_ids =["1ycr","1pw2", "2f6f", "4f5t", "1s4u", "2am9", "3my5_a", "3w8m", "4ic8"]
path = "../data/maps/"
frag_names = ["apolar", "hbacc","hbdon", "meoo", "acec", "mamn"]

idx = 8 # pdb id
pdb_id = pdb_ids[idx]
path_list = [path+pdb_id+"." + i + ".gfe.map" for i in frag_names]

for i in range(len(frag_names)):
	chan_id = i
	res, n_cells, dens, center = read_map(path_list[chan_id])
	freq_list = get_bin_frequency(dens)
	fs = 10
	plt.subplot(str(23)+str(i))
	if i==0:
		plt.xlabel('Bin Number',fontsize=fs)
		plt.ylabel('Frequency',fontsize=fs)
	sns.barplot(np.arange(0,5), freq_list, palette="deep")
	plt.title(pdb_id+" "+frag_names[chan_id],fontsize=fs)	

plt.show()
