import os

import numpy as np
from pandas import read_csv
import pandas as pd

import matplotlib.pyplot as plt
import seaborn as sns

from GSA_library.file_utils import read_ionic_output

def plot_circadapt_output(basefolder,
                        start_sample,
                        last_sample,
                        figname=None,
                        figsize=(10,5),
                        BCL=850,
                        nbeats=10,
                        mask=[]):

    """
    Plot chambers pressure-volume loops from circadapt simulations.

    Args:
        - basefolder: containing all simulations, folder numbered from
                    start_sample to last_sample
        - first_sample: first simulation number
        - last_sample: last simulation number
        - figname: path to output figure. If None, the plot is shown
        - figsize: tuple, (width,height)
        - BCL: cycle length in milliseconds, used to extract last last_beat
        - nbeats: number of simulated beats 
        - mask: boolean mask containing which simulations terminated successfully
                and which ones didn't

    """

    start = BCL*(nbeats-1)
    end = BCL*nbeats

    chambers = ['LV','RV','LA','RA']

    ax = plt.figure(figsize=figsize, constrained_layout=True).subplots(1, 4)
    for i in range(start_sample,last_sample+1):

        if mask[i]:

            for j,c in enumerate(chambers):   

                if not os.path.exists(basefolder+'/'+str(i)+'/cav.'+c+'.csv'):
                    raise Exception('Cannot find simulation file. The folder structure needs to be basefolder/i/cav.'+c+'.csv')

                ch = read_csv(basefolder+'/'+str(i)+'/cav.'+c+'.csv', delimiter=",", skipinitialspace=True,
                             	header=0, comment='#')  

                last_beat = np.where(np.array(ch['Time'])>=start)[0]
                volume = np.array(ch['Volume'][last_beat])
                pressure = np.array(ch['Pressure'][last_beat])
                t = np.array(ch['Time'][last_beat])  

                ax[j].plot(volume,pressure,color='#3489eb')  

                ax[j].set_xlabel('Volume [mL]')
                ax[j].set_ylabel('Pressure [mmHg]')
                ax[j].set_title(c)

    if figname is None:
        plt.show()
    else:
        plt.savefig(figname,dpi=100)

def plot_ionic_output(basefolder,
                    N,
                    figname=None,
                    figsize=(10,5)):

    """
    Plot ionic model output Vm and calcium.

    Args:
        - basefolder: containing all simulations, folder numbered from
                0 to N-1
        - N: number of simulations to plot
        - figname: path to output figure. If None, the plot is shown
        - figsize: tuple, (width,height)

    """
    ax = plt.figure(figsize=figsize, constrained_layout=True).subplots(1, 2)
    for i in range(N):
        if not os.path.exists(basefolder+'/'+str(i)+'/Vm.dat'):
            raise Exception('Cannot find output file. The folder structure needs to be basefolder/i/Vm.dat and Ca_i.dat')
        
        Vm = read_ionic_output(basefolder+'/'+str(i)+'/Vm.dat')
        Ca_i = read_ionic_output(basefolder+'/'+str(i)+'/Ca_i.dat')

        t = np.arange(0,Ca_i.shape[0])

        ax[0].plot(t,Vm,color='#3489eb')
        ax[1].plot(t,Ca_i,color='#3489eb')

        ax[0].set_xlabel('Time [ms]')
        ax[1].set_xlabel('Time [ms]')

        ax[0].set_ylabel('Vm [mV]')
        ax[1].set_ylabel('Ca_i [um]')

    if figname is None:
        plt.show()
    else:
        plt.savefig(figname,dpi=100)

def plot_Land_output(basefolder,
                     N,
                     figname=None,
                     isometric=False,
                     figsize=(10,5),
                     mask=[],
                     default='',
                     color='#3489eb'):

    """
    Plot Land output calcium and tension.

    Args:
        - basefolder: containing all simulations, folder numbered from
                    0 to N-1
        - N: number of simulations to plot
        - figname: path to output figure. If None, the plot is shown
        - isometric: if True, no stretch is plotted
        - figsize: tuple, (width,height)
        - mask: boolean mask containing which simulations terminated successfully
                and which ones didn't
        - default: path to folder with Land output (Tension.dat for instance) 
                   that can be plotted against the simulations for comparison

    """

    if not isometric:
        ax = plt.figure(figsize=figsize, constrained_layout=True).subplots(1, 3)
    else:
        ax = plt.figure(figsize=figsize, constrained_layout=True).subplots(1, 2)

    plot_all = np.arange(N)
    if len(mask)>0:
        plot_idx = plot_all[np.where(mask==1)[0]]
    else:
        plot_idx = plot_all

    for i in plot_all:
        if not os.path.exists(basefolder+'/'+str(i)+'/Tension.dat'):
            raise Exception('Cannot find output file. The folder structure needs to be basefolder/i/Tension.dat and Ca_i.dat')

        T = read_ionic_output(basefolder+'/'+str(i)+'/Tension.dat')
        Ca_i = read_ionic_output(basefolder+'/'+str(i)+'/Ca_i.dat')
        t = np.arange(0,Ca_i.shape[0])

        if i in plot_idx:
            if np.max(np.abs(T))<500.0:
                ax[0].plot(t,Ca_i,color=color)
                ax[1].plot(t,T,color=color)
        else:
            if np.max(np.abs(T))<500.0:
                ax[0].plot(t,Ca_i,color='black',zorder=0)
                ax[1].plot(t,T,color='black',zorder=0)

        ax[0].set_xlabel('Time [ms]')
        ax[1].set_xlabel('Time [ms]')

        ax[0].set_ylabel('Ca_i [um]')
        ax[1].set_ylabel('Tension [kPa]')

        # lambda_out = read_ionic_output(basefolder+'/'+str(i)+'/lambda.dat')
        if not isometric:
            lambda_out = read_ionic_output(basefolder+'/'+str(i)+'/stretch.dat')
            if i in plot_idx:
                ax[2].plot(t,lambda_out,color=color)
            else:
                ax[2].plot(t,lambda_out,color='black',zorder=0)
            ax[2].set_xlabel('Time [ms]')
            ax[2].set_ylabel('Lambda [-]')

        if default != '':
            T = read_ionic_output(default+'/Tension.dat')
            Ca_i = read_ionic_output(default+'/Ca_i.dat')
            
            ax[0].plot(t,Ca_i,'--',
                      color='black',
                      linewidth=2.0
                      )
            ax[1].plot(t,T,'--',
                      color='black',
                      linewidth=2.0
                      )

    if figname is None:
        plt.show()
    else:
        plt.savefig(figname,dpi=100)

def plot_dataset(Xdata, Ydata, xlabels, ylabels, savepath, figsize=(9,6)):
    """
    Plot Y high-dimensional dataset by pairwise plotting its features against each X dataset's feature.

    Args:
            - Xdata: n*m1 matrix
            - Ydata: n*m2 matrix
            - xlabels: list of m1 strings representing the name of X dataset's features
            - ylabels: list of m2 strings representing the name of Y dataset's features
            
    """

    sample_dim = Xdata.shape[0]
    in_dim = Xdata.shape[1]
    out_dim = Ydata.shape[1]
    fig, axes = plt.subplots(
        nrows=out_dim,
        ncols=in_dim,
        sharex="col",
        sharey="row",
        figsize=figsize,
    )
    for i, axis in enumerate(axes.flatten()):
        axis.scatter(
            Xdata[:, i % in_dim], Ydata[:, i // in_dim], c='#3489eb', s=0.1
        )
        inf = min(Xdata[:, i % in_dim])
        sup = max(Xdata[:, i % in_dim])
        mean = 0.5 * (inf + sup)
        delta = sup - mean
        if i // in_dim == out_dim - 1:
            axis.set_xlabel(xlabels[i % in_dim],rotation=90)
            axis.set_xticks([])
            axis.set_xlim(left=inf - 0.3 * delta, right=sup + 0.3 * delta)
        if i % in_dim == 0:
            axis.set_yticks([])
            axis.set_ylabel(ylabels[i // in_dim])
    plt.suptitle("Sample dimension = {} points".format(sample_dim))
    plt.savefig(savepath + "X_vs_Y.png", bbox_inches="tight", dpi=300)

