import numpy as np
import sys
import os
sys.path.append(os.path.join(os.path.dirname(__file__), ".."))
from src.mapIO import read_map
import matplotlib.pyplot as plt
from matplotlib import mlab
from src.util import box_face_ave


def scatter_plot(pdb_id, frag_names, frag_names_short, path, pred_path, title):
    '''
    Scatter plot of measured vs predicted GFE per fragmap type, per protein.
    input: list of proteins (with already generated fragname_o.gfe.map file generated
    output: 2x3 multi scatterplot saved as pdb_id.png, saved to /FragMaps_CNN/figs
    '''

    print('plotting ', pdb_id, ' scatterplot')
    pred_names_short = [i+'P' for i in frag_names_short]
    tail = ".gfe.map"
    path_list = [path+pdb_id+"."+name+tail for name in frag_names_short]
    pred_path_list = [pred_path+pdb_id+"."+name+tail for name in pred_names_short]

    fig = plt.figure(figsize=(20,10))
    plot_fn= pdb_id +'_scatter.png'

    ## CHECK TITLE BEFORE PLOTTING
    plt.title(pdb_id + "  predicted vs measured GFE scatter")
    plt.axis('off')

    for i in range(1,len(path_list)+1):
        _, _, gfe_p, _ = read_map(pred_path_list[i-1])
        _, _, gfe, _ = read_map(path_list[i-1])

        x = gfe_p
        y = gfe

        ax = fig.add_subplot(2, 3, i)
        colors = 'bgrcmy'



        l_x = np.arange(-3.0, 3.0, 0.1)
        ax.plot(l_x,l_x, color='black')
        frag = frag_names[i-1]

        plt.scatter(x, y, s=0.01, color=colors[i-1])

        plt.xlim(-2.2, 2.2)
        plt.ylim(-2.2, 2.2)

        plt.grid()
        plt.legend([frag], loc="best")
        plt.ylabel("y=GFE measured")
        plt.xlabel("x=GFE predicted")
        


    fig.set_tight_layout(True)
    fig.savefig('figs/'+plot_fn)

    return


if __name__=='__main__':

    
    pdb_ids = ["1ycr", "1pw2", "2f6f", "4ic8",
               "1s4u", "2am9", "2zff", "1bvi_m1_fixed",
               "4djw", "4lnd_ba1", "4obo", "1h1q",
               "3fly",  "4gih", "2gmx", "4hw3",
               "3w8m", "2qbs",  "4jv7", "4ypw_prot_nocter",
               "5q0i",  "3my5_a",  "1r2b", "2jjc",
               "3bi0", "4f5t", "4wj9", "1d6e_nopep_fixed"]       

    
    frag_names = ["apolar", "hbacc","hbdon", "meoo", "acec", "mamn"]
    
    mpath = "/home/tr443/Projects/Frag_Maps/data/maps/"
    ppath = "../output/maps/"
    
    
    for ipdb in pdb_ids:

        scatter_plot(ipdb, frag_names, frag_names, path = mpath, pred_path = ppath, title = ipdb)
