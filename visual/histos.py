import numpy as np
import sys
import os
sys.path.append(os.path.join(os.path.dirname(__file__), ".."))
from src.mapIO import read_map
from src.util import box_face_ave
import matplotlib.pyplot as plt

frag_names = ["Gen. Apolar","Gen. Acceptor",
              "Gen. Donor","Methanol Oxy",
              "Acec", "Methylammonium Nitro"]

frag_names_short = ["apolar", "hbacc", "hbdon", "meoo","acec", "mamn"]

path = "../data/maps/"
pdb_ids = ["1ycr", "1pw2", "2f6f", "4f5t", "1s4u", "2am9", "3my5_a", "3w8m","4ic8"]
pdb_id = pdb_ids[7]

tail = ".gfe.map"
path_list = [path+pdb_id+"."+name+tail for name in frag_names_short]


leg = []
kBT = 0.592 # T=298K, kB = 0.001987 kcal/(mol K) 
for i in range(len(path_list)):
    _, _, gfe,_ = read_map(path_list[i])

    f_ave = box_face_ave(gfe)
    #print(f_ave, np.median(gfe))

    #gfe[gfe > 2.50] = 0.0
    new_shape = gfe.shape[0]*gfe.shape[1]*gfe.shape[2]
    vec = np.reshape(gfe,new_shape) - f_ave
    #vec = np.exp(vec/kBT)  
    
    plt.hist(vec, histtype='barstacked', bins = 200, alpha = 0.4)
    mean = round(vec.mean(),2)
    #print(mean)
    leg.append(frag_names[i] + ",  " + r"$\mu$ =" + str(mean))    

plt.title(pdb_id)
plt.legend(leg, loc = "best", frameon = False)
plt.grid()
plt.xlabel("GFE")
plt.ylabel("Freq")
plt.show()
