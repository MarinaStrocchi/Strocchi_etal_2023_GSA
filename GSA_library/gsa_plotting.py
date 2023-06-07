import re
import math

import numpy as np
from pandas import read_csv
import pandas as pd

from matplotlib import cm
import matplotlib.pyplot as plt
import matplotlib.gridspec as grsp
from mpl_toolkits.axes_grid1.inset_locator import inset_axes
import seaborn as sns

from gpytGPE.utils.design import read_labels
from Historia.history import hm

from GSA_library.file_utils import read_ionic_output
from GSA_library.gsa_parameters_ranking import correct_S

# sns.reset_orig()

def gsa_heat(ST, 
             S1,
             xlabels, 
             ylabels, 
             savepath, 
             height=10,
             width=10, 
             correction=False, 
             horizontal=False, 
             cbar_shift=None,
             cbar_width=10,
             xlabels_latex=None,
             ylabels_latex=None,
             fontsize=16):

    """
    Plots a heatmap for a global sensitivity analysis.

    Args:
        - ST: total order effects matrix 
        - S1: first order effects matrix 
        - xlabels: parameter names
        - ylabels: output names
        - savepath: where to save the figure
        - height: figure height
        - width: figure width
        - correction: if you want to get rid of very small effects (below 0.01)
        - horizontal: if you want an horizontal heatmap or not
        - cbar_shift: shift to apply to the colorbar
        - cbar_width: how wide should your colorbar be (set to 0 to get rid of the colorbar)
        - xlabels_latex: latex parameter names - for paper plotting
        - ylabels_latex: latex output names - for paper plotting
        - fontsize: size of figure font

    """

    if correction:
        ST = correct_S(ST, 0.01)
        S1 = correct_S(S1, 0.01)

    ST = ST/np.sum(ST,axis=0)

    plt.rc('text', usetex=False)

    if cbar_shift is None:
        if not horizontal:
            cbar_shift = 0.0
        else:
            cbar_shift = 0.0

    if not horizontal:
        fig, axes = plt.subplots(1, 2, figsize=(width,height))
        df = pd.DataFrame(data=S1, index=xlabels, columns=ylabels)
        rot_angle_x = 45
        rot_angle_y = 45
        cbar_size = 0.8
        cbar_orientation = "vertical"

        cax = inset_axes(axes[0],
                 width=str(cbar_width)+"10%",  
                 height="100%",  
                 loc='center right',
                 )
        cbar_args={"shrink": cbar_size,"label": "Sensitivity","orientation": cbar_orientation}
        cbar_bool = True

        if cbar_width==0:
            cbar_bool = False

    else:
        fig, axes = plt.subplots(2, 1, figsize=(height,width))
        df = pd.DataFrame(data=np.transpose(S1), index=ylabels, columns=xlabels)

        rot_angle_x = 90
        rot_angle_y = 0
        cbar_size = 0.8
        cbar_orientation = "vertical"

    if cbar_width>0:
        cbar_args={"shrink": cbar_size,"label": "Sensitivity","orientation": cbar_orientation}

        h1 = sns.heatmap(
            df,
            cmap="rocket_r",
            vmin=0.0,
            vmax=1.0,
            square=True,
            linewidth=0.5,
            linecolor='black',
            cbar=True,
            cbar_kws=cbar_args,
            ax=axes[0],
        )

    else:
        h1 = sns.heatmap(
            df,
            cmap="rocket_r",
            vmin=0.0,
            vmax=1.0,
            square=True,
            linewidth=0.5,
            linecolor='black',
            cbar=False,
            ax=axes[0],
        )

    h1.set_yticks(np.arange(df.shape[0])+0.5)
    axes[0].set_title("S1", fontsize=12, fontweight="bold")
    axes[0].tick_params(left=False, bottom=False)
    if xlabels_latex is None:
        h1.set_xticklabels(h1.get_xticklabels(), rotation=rot_angle_x, va="top",fontsize=fontsize)
    else:
        h1.set_xticklabels(xlabels_latex, rotation=rot_angle_x, va="top",fontsize=fontsize)
    
    if ylabels_latex is None:
        h1.set_yticklabels(h1.get_yticklabels(), rotation=rot_angle_y, ha="right",fontsize=fontsize)
    else:
        h1.set_yticklabels(ylabels_latex, rotation=rot_angle_y, va="top",fontsize=fontsize)

    if not horizontal:
        df = pd.DataFrame(data=ST, index=xlabels, columns=ylabels)
        img_name = "heatmap.png"
    else:
        df = pd.DataFrame(data=np.transpose(ST), index=ylabels, columns=xlabels)
        img_name = "heatmap_horizontal.png"

    ht = sns.heatmap(
        df,
        cmap="rocket_r",
        vmin=0.0,
        vmax=1.0,
        square=True,
        linewidth=0.8,
        linecolor='black',
        cbar=True,
        cbar_kws=cbar_args,
        ax=axes[1],
    )
    axes[1].set_title("ST", fontsize=12, fontweight="bold")
    axes[1].tick_params(left=False, bottom=False)

    ht.set_yticks(np.arange(df.shape[0])+0.5)
    if xlabels_latex is None:
        ht.set_xticklabels(ht.get_xticklabels(), rotation=rot_angle_x, va="top",fontsize=fontsize)
    else:
        ht.set_xticklabels(xlabels_latex, rotation=rot_angle_x, va="top",fontsize=fontsize)
    
    if ylabels_latex is None:
        ht.set_yticklabels(ht.get_yticklabels(), rotation=rot_angle_y, ha="right",fontsize=fontsize)
    else:
        ht.set_yticklabels(ylabels_latex, rotation=rot_angle_y, va="top",fontsize=fontsize)

    if (xlabels_latex is not None) or (ylabels_latex is not None):
        plt.rc('text', usetex=False)
        # plt.rc('font', family='serif')
    else:
        plt.rc('text', usetex=False)
        
    plt.tight_layout()
    plt.savefig(savepath + img_name, bbox_inches="tight", dpi=1000)

def plot_rank_land_comparison(loadpath,
                              features,
                              plot_labels,
                              criterion = "STi",
                              figname=""):

    """
    Plots the parameter ranking for the land model for different simulation
    types, for instance isometric with different stretches and isotonic.

    Args:
        - loadpath: list of folders containing the GPEs and GSA output folders
        - features: list of features to consider 
        - plot_labels: list of legend labels
        - criterion: STi or Si e.g. total or first order effects to rank the parameters
        - figname: name of the figure

    """

    colors = ["#34abeb","#ff8000","#1bab1b"]

    # assumes that xlabels and ylabels are the same for all tests
    index_i = read_labels(loadpath[0] + "data/xlabels.txt")
    label = read_labels(loadpath[0] + "data/ylabels.txt")

    x = np.arange(len(index_i))
    barWidth = 0.25

    fig, ax = plt.subplots(1, 1, figsize=(15,5), constrained_layout=True)

    for p,path in enumerate(loadpath):
        
        if criterion == "Si":
            tag = "first-order"
        elif criterion == "STi":
            tag = "total"

        r_dct = {key: [] for key in index_i}    

        for idx in features:

            S = np.loadtxt(path + "output/" + f"{idx}/" + str() + criterion + ".txt", dtype=float)
            # S = correct_S(S, 0.01)

            mean = np.array([S[:, i].mean() for i in range(len(index_i))])
            ls = list(np.argsort(mean))

            for i, idx in enumerate(ls):
                if mean[idx] != 0:
                    r_dct[index_i[idx]].append(i)
                else:
                    r_dct[index_i[idx]].append(0)

        bars = []
        for key in r_dct.keys():
            bars.append(np.sum(r_dct[key]))

        r = [xx + p*barWidth for xx in x]

        ax.bar(r, bars, color=colors[p], width=barWidth, edgecolor='white', label=plot_labels[p])
    
    plt.legend()
    plt.xticks(x+barWidth, index_i, rotation=90)

    if figname == "":
        plt.show()
    else:
        plt.savefig(figname)

def plot_rank_land_comparison_S(loadpath,
                                plot_labels,
                                criterion = "STi",
                                mode="max",
                                figname="",
                                th=0.0,
                                sort=False,
                                xlabels_latex=None,
                                fontsize=18):

    """
    Plots the parameter ranking for the land model for different simulation
    types, for instance isometric with different stretches and isotonic.

    Args:
        - loadpath: list of folders containing the GPEs and GSA output folders
        - plot_labels: list of legend labels
        - criterion: STi or Si e.g. total or first order effects to rank the parameters
        - mode: maximum or mean - how to rank the parameteres across simulations
        - figname: name of the figure
        - th: threshold on the cumulative effect where to draw a line between
              important and unimportant parameters
        - sort: if you want to sort the parameters (True/False)
        - xlabels_latex: latex parameter names for paper plots
        - fontsize: size of figure font
    """

    colors = ["#34abeb","#ff8000","#1bab1b"]

    # if xlabels_latex is not None:
    #     plt.rc('text', usetex=True)
    #     plt.rc('font', family='serif')
    # else:
    #     plt.rc('text', usetex=False)

    # assumes that xlabels and ylabels are the same for all tests
    index_i = read_labels(loadpath[0] + "data/xlabels.txt")
    label = read_labels(loadpath[0] + "data/ylabels.txt")

    x = np.arange(len(index_i))
    barWidth = 0.25

    bars_all = np.zeros((len(index_i),len(loadpath)),dtype=float)
    for p,path in enumerate(loadpath):
        
        if criterion == "Si":
            tag = "first-order"
        elif criterion == "STi":
            tag = "total"

        f = open(path+"output/Rank_"+criterion+"_"+mode+".txt","r")
        lines = f.readlines()

        r_dct = {}
        for line in lines:
            line_split = re.split(r'\t+', line)
            r_dct[line_split[0]] = float(line_split[1])

        bars = []
        for l in index_i:
            bars.append(r_dct[l])
        bars_all[:,p] = bars

    if sort:
        idx_sorted = list(np.argsort(np.max(bars_all,axis=1)))

        if xlabels_latex is None:
            index_i_sorted = [index_i[idx] for idx in idx_sorted]
        else:
            index_i_sorted = [xlabels_latex[idx] for idx in idx_sorted]

        idx_sorted = idx_sorted[::-1]
        index_i_sorted = index_i_sorted[::-1]
    else:
        idx_sorted = range(len(index_i))

        if xlabels_latex is None:
            index_i_sorted = index_i
        else:
            index_i_sorted = xlabels_latex

    bars_sorted = np.max(bars_all,axis=1)
    bars_sorted = bars_sorted[idx_sorted]
    bars_sorted_norm = list(bars_sorted/sum(bars_sorted))

    bars_sorted_sum = []
    for i in range(len(bars_sorted)):
        bars_sorted_sum.append(sum(bars_sorted_norm[0:i+1]))

    print(bars_sorted_sum)

    f = open("./Rank_"+criterion+"_"+mode+"_all.txt", "w")
    for i in range(len(index_i)):
        f.write(index_i[idx_sorted[i]]+"\t"+str(bars_sorted[i])+"\n")
    f.close()

    f = open("./Rank_"+criterion+"_"+mode+"_all_ExpVariance.txt", "w")
    for i in range(len(index_i)):
        f.write(index_i[idx_sorted[i]]+"\t"+str(bars_sorted_norm[i])+"\t"+str(bars_sorted_sum[i])+"\n")
    f.close()

    fig, ax = plt.subplots(1, 1, figsize=(15,5), constrained_layout=True)

    for p,path in enumerate(loadpath):

        r = [xx + p*barWidth for xx in x]

        ax.bar(r, bars_all[idx_sorted,p], color=colors[p], width=barWidth, edgecolor='white', label=plot_labels[p])

    if th > 0:
        ax.plot([-2*barWidth,len(index_i)+2*barWidth],[th,th],color='black',linestyle='--')

    if criterion=='STi':
        ax.set_ylabel('$ST$',fontsize=fontsize)
    else:
        ax.set_ylabel('$S_1$',fontsize=fontsize)

    plt.legend()
    plt.xticks(x+barWidth, index_i_sorted, rotation=90,fontsize=16)

    y_max = np.ceil(np.max(np.array(bars_all.flatten()))*10)/10
    yticks = np.arange(0,y_max+0.05,0.05, dtype=float)

    cutoff = r[np.where(np.array(bars_sorted_sum)>0.9)[0][0]]+barWidth
    ax.plot([cutoff,cutoff],[0,y_max],color='black',linestyle='--')

    ax.tick_params(axis='y', labelsize=16)
    # plt.ylabel('Sensitivity',fontsize=16)
    
    if figname == "":
        plt.show()
    else:
        plt.savefig(figname)

def plot_rank_GSA(datapath,
                  loadpath,
                  rank_file=None,
                  criterion="STi",
                  mode="max",
                  figname="",
                  normalise=False,
                  th=0.0,
                  annotate=False,
                  figsize=(15,5),
                  fontsize=14,
                  xlabels_latex=None,
                  separate_colors=False,
                  color_important=None):

    """
    Plots the parameter ranking for a .

    Args:
        - datapath: folder with data (xlabels.txt, etc...)
        - loadpath: path where you saved your parameter ranking 
        - rank_file: if you want to provide a different parameter ranking file 
                     that is not in the loadpath folder
        - criterion: STi or Si e.g. total or first order effects to use for ranking
        - mode: max or mean to rank the parameters
        - figname: name of output figure 
        - normalise: if you want to normalise so that the values all sum up to 1 
        - th: threshold to determine which parameters are important and which ones aren't
        - annotate: write numbers on top of each bar
        - figsize: size of output figure
        - fontsize: size of figure font
        - xlabels_latex: parameter names in latex for paper plots
        - separate_colors: if you want a different colour for important and unimportant parameters
        - color_important: what colour you want the important parameter bars to be
    """

    color = ["#ff8000"]

    # assumes that xlabels and ylabels are the same for all tests
    index_i = read_labels(datapath + "/xlabels.txt")

    x = np.arange(len(index_i))
    barWidth = 0.25

    fig, ax = plt.subplots(1, 1, figsize=figsize, constrained_layout=True)
        
    if criterion == "Si":
        tag = "first-order"
    elif criterion == "STi":
        tag = "total"

    if rank_file is None:
        rank_file = loadpath+"/Rank_"+criterion+"_"+mode+".txt"

    f = open(rank_file,"r")
    lines = f.readlines()

    r_dct = {}
    for line in lines:
        line_split = re.split(r'\t+', line)
        r_dct[line_split[0]] = float(line_split[1])

    bars = []
    for l in index_i:
        bars.append(r_dct[l])
    idx_sorted = np.argsort(np.array(bars))

    r = [xx + barWidth for xx in x]

    bars_sorted = [bars[idx] for idx in idx_sorted]
    bars_sorted = bars_sorted[::-1]
    if xlabels_latex is not None:
        index_i_sorted = [xlabels_latex[idx] for idx in idx_sorted]
    else:
        index_i_sorted = [index_i[idx] for idx in idx_sorted]
    index_i_sorted = index_i_sorted[::-1]

    bars_sorted_norm = list(np.array(bars_sorted)/sum(bars_sorted))

    bars_sorted_sum = []
    for i in range(len(bars_sorted)):
        bars_sorted_sum.append(sum(bars_sorted_norm[0:i+1]))
    print(bars_sorted_sum)

    if normalise:
        barplot = bars_sorted_norm
    else:
        barplot = bars_sorted

    plt.xticks(x+barWidth, index_i_sorted, rotation=90,fontsize=fontsize)
    ax.tick_params(axis='both',labelsize=fontsize)

    cutoff_param = np.where(np.array(bars_sorted_sum)>0.9)[0][0]
    if color_important is None:
        color_important = "darkorange"
    color_unimportant = "lightgray"
    colors = [color_important,]*(cutoff_param+1)+[color_unimportant,]*(len(bars_sorted_sum)-cutoff_param-1)

    if separate_colors:
        bars = ax.bar(r, barplot, width=barWidth, edgecolor='white',color=colors)
    else:
        bars = ax.bar(r, barplot, color=color, width=barWidth, edgecolor='white')

    # if xlabels_latex is not None:
    #     plt.rc('text', usetex=False)
    #     # plt.rc('font', family='serif')
    # else:
    #     plt.rc('text', usetex=False)
    if criterion=='STi':
        ax.set_ylabel('$ST$',fontsize=fontsize)
    else:
        ax.set_ylabel('$S_1$',fontsize=fontsize)
    
    if th > 0:
        ax.plot([-2*barWidth,len(index_i)+2*barWidth],[th,th],color='black',linestyle='--')
    
    plt.legend()

    y_max = np.ceil(np.max(np.array(barplot))*10)/10
    yticks = np.arange(0,y_max+0.05,0.05, dtype=float)



    cutoff = r[cutoff_param]+2*barWidth
    ax.plot([cutoff,cutoff],[0,y_max],color='black',linestyle='--')

    if annotate:
        for i,bar in enumerate(bars):
            yval = bar.get_height()
            ax.text(bar.get_x(), yval + .005, str(round(bars_sorted_sum[i]*100))+'%',fontsize=fontsize)

    # ax.set_yticks(yticks)
    # ax.set_yticklabels(np.round(yticks,2),fontsize=16)

    if figname == "":
        plt.show()
    else:
        plt.savefig(figname)