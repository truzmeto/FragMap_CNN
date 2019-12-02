import numpy as np
import sys
import os
sys.path.append(os.path.join(os.path.dirname(__file__), ".."))
from src.mapIO import read_map 
import matplotlib.pyplot as plt
from matplotlib import mlab
from src.util import box_face_ave


frag_names = ["Gen. Apolar","Gen. Acceptor",
              "Gen. Donor","Methanol Oxy",
              "Acec", "Methylammonium Nitro"]

frag_names_short = ["apolar", "hbacc", "hbdon", "meoo","acec", "mamn"]

path = "../data/maps/"
#pdb_id = "1ycr"
pdb_id = "1pw2"
tail = ".gfe.map"
path_list = [path+pdb_id+"."+name+tail for name in frag_names_short]



leg = []
nbins = 200
for i in range(len(path_list)):
    _, _, gfe = read_map(path_list[i])
    
    #gfe[gfe> 3.30] = 0.0
    f_ave = box_face_ave(gfe)
    new_shape = gfe.shape[0]*gfe.shape[1]*gfe.shape[2]
    vec = np.reshape(gfe,new_shape) - f_ave
    
    #plt.hist(vec, histtype='barstacked', bins = 60, alpha=0.4)
    values, base = np.histogram(vec, bins = nbins)
    cum = np.cumsum(values)
    plt.plot(base[:-1], cum)
        
    #mean = round(np.mean(vec),2)
    median = round(np.median(vec),2)
    
    leg.append(frag_names[i]+ ",  " + "median=" + str(median))
    
plt.title(pdb_id+", nbins = "+str(nbins))
plt.legend(leg, loc="best")
plt.xlabel("GFE")
plt.ylabel("Freq")
plt.show()
