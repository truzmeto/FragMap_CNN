import numpy as np
import sys
import os
import matplotlib.pyplot as plt
from matplotlib import mlab

sys.path.append(os.path.join(os.path.dirname(__file__), ".."))
from src.mapIO import read_map 
from src.util import grid2vec

pdb_ids = ["1ycr","1pw2","2f6f", "4f5t", "2am9", "3my5_a", "3w8m", "4ic8"] 
path = "../output/maps/"
frag_names = ["apolar", "hbacc","hbdon", "meoo", "acec", "mamn"]
idx = 3 # pdb id
idx_map = 0# map id
pdb_id = pdb_ids[idx]

#path_list = [path+pdb_id+"." + i + ".gfe.map" for i in frag_names]
path_list = [path + pdb_id+"." + frag_names[idx_map] + ".gfe.map"]
path_listP = [path + pdb_id+"." + frag_names[idx_map]+"P"+".gfe.map"]

_, _, gfe, _ = read_map(path_list[idx_map])
_, _, gfeP, _ = read_map(path_listP[idx_map])

orig = grid2vec(gfe.shape, gfe)
pred = grid2vec(gfeP.shape, gfeP)

fig = plt.figure()
plt.scatter(pred, orig)
#plt.plot(orig, pred, ".")

plt.legend(frag_names[idx_map], loc = "best")
plt.ylabel("y=GFE measured")
plt.xlabel("x=GFE predicted")

fig.set_tight_layout(True)
plt.show()
#fig.savefig('../figs/'+plot_fn)

