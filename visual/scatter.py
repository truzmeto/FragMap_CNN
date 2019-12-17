import numpy as np
import sys
import os
sys.path.append(os.path.join(os.path.dirname(__file__), ".."))
from src.mapIO import read_map 
import matplotlib.pyplot as plt
from matplotlib import mlab
from src.util import box_face_ave


pdb_list = ['1ycr', '4ic8']


def scatter_plot(pdb_id):
    '''
    Scatter plot of measured vs predicted GFE per fragmap type, per protein.
    input: list of proteins (with already generated fragname_o.gfe.map file generated
    output: 2x3 multi scatterplot saved as pdb_id.png, saved to /FragMaps_CNN/figs
    '''
    
    frag_names = ["Gen. Apolar","Gen. Acceptor",
                "Gen. Donor","Methanol Oxy",
                "Acec", "Methylammonium Nitro"]

    frag_names_short = ["apolar", "hbacc", "hbdon", "meoo","acec", "mamn"]

    pred_names_short = [i+'_o' for i in frag_names_short]

    #print(pred_names_short)

    path = "../data/maps/"
    pred_path = "../output/"

    tail = ".gfe.map"

    path_list = [path+pdb_id+"."+name+tail for name in frag_names_short]

    pred_path_list = [pred_path+pdb_id+"."+name+tail for name in pred_names_short]

    #print(pred_path_list)


    fig = plt.figure(figsize=(20,10))
    plot_fn= pdb_id +'_scatter.png'


    for i in range(1,len(path_list)+1):
        _, _, gfe, _ = read_map(path_list[i-1])
        _, _, gfe_p, _ = read_map(pred_path_list[i-1])
        
        #print(gfe.shape)
        #print(gfe_p.shape)
        
        #print(gfe)
        #print(gfe_p)
        

        x = gfe
        y = gfe_p

        ax = fig.add_subplot(2, 3, i)

        # Move right y-axis and top x-axis to center
        ax.spines['right'].set_position('center')
        ax.spines['top'].set_position('center')

        # Eliminate upper and right axes
        ax.spines['right'].set_color('red')
        ax.spines['top'].set_color('green')

        # Show ticks in the left and lower axes only
        ax.xaxis.set_ticks_position('top')
        ax.yaxis.set_ticks_position('right')
        
        ax.xaxis.set_label_coords(0.5, 0)
        ax.yaxis.set_label_coords(0, 0.5)
        
        l_x = np.arange(np.minimum(x.min(), y.min()),np.maximum(x.max(), y.max()),0.1)
        #print(l_x, l_y)
        
        plt.plot(l_x,l_x, color='black')
        
        frag = frag_names[i-1]

        plt.grid()
        plt.scatter(x, y, s=0.01)
        plt.title(pdb_id+" predicted vs measured GFE scatter")
        plt.legend([frag,], loc="best")
        plt.ylabel("y=GFE measured")
        plt.xlabel("x=GFE predicted")




    fig.set_tight_layout(True)
    #plt.show()
    fig.savefig('../figs/'+plot_fn)

    return


for i in pdb_list:
    print('plotting protein ', i)
    scatter_plot(i)
    


