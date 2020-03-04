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

maps = ["apolar", "hbacc", "hbdon", "meoo","acec", "mamn"]

path = "/u1/home/tr443/data/fragData/maps/"
pdb_ids = ["3my5_a", "4f5t", "4wj9", "3bi0", "1d6e_nopep_fixed",
           "1ycr", "1pw2", "2f6f", "4ic8", "1s4u", "2am9",
           "1bvi_m1_fixed", "4djw", "4lnd_ba1", "4obo",
           "1h1q", "3fly", "4gih", "2gmx", "4hw3", "4ypw_prot_nocter",
            "3w8m", "2qbs", "4jv7", "5q0i", "1r2b", "2jjc"]

tail = ".gfe.map"
#gfe_med = []
#f_med = []

for ipdb in pdb_ids:
    for imap in maps:

        map_path = path + ipdb + "." + imap + tail
        _, _, gfe,_ = read_map(map_path)

        face_med = box_face_med(gfe)
        box_med = np.median(gfe)
        #print(ipdb, imap, "Mean",        gfe.mean())
        print(ipdb, imap, "Median",      box_med )
        print(ipdb, imap, "Face Median", round(face_med,2))

        #gfe_med.append(box_med)
        #f_med.append(face_med)
        
        
#plt.plot(gfe_med, f_med, ".")
#plt.show()
        
        
        
