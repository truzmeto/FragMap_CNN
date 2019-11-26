import numpy as np
import sys
import os
sys.path.append(os.path.join(os.path.dirname(__file__), ".."))
from src.mapIO import read_map 
import matplotlib.pyplot as plt

frag_names = ["Gen. Apolar","Gen. Acceptor",
              "Gen. Donor","Methanol Oxy",
              "Acec", "Methylammonium Nitro"]

frag_names_short = ["apolar", "hbacc", "hbdon", "meoo","acec", "mamn"]

path = "../data/maps/"
pdb_id = "1ycr"
tail = ".gfe.map"
path_list = [path+pdb_id+"."+name+tail for name in frag_names_short]


leg = []
for i in range(len(path_list)):
    _, _, dens = read_map(path_list[i])

    #dens[dens > 2.50] = 0.0
    new_shape = dens.shape[0]*dens.shape[1]*dens.shape[2]
    vec = np.reshape(dens,new_shape)
    plt.hist(vec, histtype='barstacked', bins = 60, alpha=0.4)

    mean = round(vec.mean(),2)
    leg.append(frag_names[i]+ ",  " + r"$\mu$ =" + str(mean))    
plt.title(pdb_id)
plt.legend(leg, loc="best")
plt.xlabel("GFE")
plt.ylabel("Freq")
plt.show()
