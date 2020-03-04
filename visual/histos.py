import numpy as np
import sys
import os
sys.path.append(os.path.join(os.path.dirname(__file__), ".."))
from src.mapIO import read_map
from src.util import box_face_ave, box_face_med
import matplotlib.pyplot as plt

frag_names = ["Gen. Apolar","Gen. Acceptor",
              "Gen. Donor","Methanol Oxy",
              "Acec", "Methylammonium Nitro"]

frag_names_short = ["apolar", "hbacc", "hbdon", "meoo","acec", "mamn"]

#path = "../data/maps/"
path = "/u1/home/tr443/data/fragData/maps/"
pdb_ids = ["3my5_a", "4f5t", "4wj9", "3bi0", "1d6e_nopep_fixed",
           "1ycr", "1pw2", "2f6f", "4ic8", "1s4u", "2am9",
           "1bvi_m1_fixed", "4djw", "4lnd_ba1", "4obo",
           "1h1q", "3fly", "4gih", "2gmx", "4hw3", "4ypw_prot_nocter",
            "3w8m", "2qbs", "4jv7", "5q0i", "1r2b", "2jjc"]


pdb_id = pdb_ids[15]
tail = ".gfe.map"
path_list = [path + pdb_id + "." + name + tail for name in frag_names_short]


leg = []
RT = 0.59248368 # T=298.15K, R = 0.001987204 kcal/(mol K)

for i in range(len(path_list)):
    _, _, gfe,_ = read_map(path_list[i])

    median = box_face_med(gfe)
    #
    #median = round(np.median(gfe),2)
    new_shape = gfe.shape[0] * gfe.shape[1] * gfe.shape[2]
    vec = np.reshape(gfe, new_shape) - median

    plt.hist(vec, histtype='barstacked', bins = 200, alpha = 0.4)
    leg.append(frag_names[i] + ",  " + r"GFE_med =" + str(median))    

plt.title(pdb_id)
plt.legend(leg, loc = "best", frameon = False)
plt.grid()
plt.xlabel("GFE")
plt.ylabel("Freq")
plt.show()
