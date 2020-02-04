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

    #print(pred_names_short)



    tail = ".gfe.map"

    path_list = [path+pdb_id+"."+name+tail for name in frag_names_short]

    pred_path_list = [pred_path+pdb_id+"."+name+tail for name in pred_names_short]

    #print(pred_path_list)


    fig = plt.figure(figsize=(20,10))
    plot_fn= pdb_id +'_'+title+'_scatter.png'

    ## CHECK TITLE BEFORE PLOTTING
    plt.title(pdb_id+' '+title+" predicted vs measured GFE scatter")

    plt.axis('off')

    for i in range(1,len(path_list)+1):
        _, _, gfe_p, _ = read_map(pred_path_list[i-1])
        _, _, gfe, _ = read_map(path_list[i-1])



        x = gfe_p
        y = gfe

        ax = fig.add_subplot(2, 3, i)

        colors = 'bgrcmy'



        xmin= x.min()
        xmax= x.max()
        ymin= y.min()
        ymax= y.max()

        absmin = np.minimum(xmin, ymin)
        absmax = np.maximum(xmax, ymax)

        l_x = np.arange(absmin,absmax,0.1)
        #print(l_x, l_y)

        ax.plot(l_x,l_x, color='black')

        frag = frag_names[i-1]



        plt.scatter(x, y, s=0.01, color=colors[i-1])

        #plt.xlim(absmin, absmax)
        #plt.ylim(absmin, absmax)
        plt.xlim(-2., 3.)
        plt.ylim(-2., 3.)

        plt.grid()
        plt.legend([frag], loc="best")
        plt.ylabel("y=GFE measured")
        plt.xlabel("x=GFE predicted")
        



    fig.set_tight_layout(True)

    #print(os.getcwd())

    #plt.show()
    fig.savefig('../figs/'+plot_fn)

    return


if __name__=='__main__':


    pdb_ids = ["1ycr","1pw2","2f6f", "4f5t", "2am9", "3my5_a", "3w8m", "4ic8"]
    frag_names = ["apolar", "hbacc","hbdon", "meoo", "acec", "mamn"]
    # path = "../output/maps/"
    path = "../output/maps/"
    scatter_plot(pdb_ids[0], frag_names, frag_names, path, path, title='')
