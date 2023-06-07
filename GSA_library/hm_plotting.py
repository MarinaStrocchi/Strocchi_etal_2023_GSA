import re
import math

import numpy as np
from pandas import read_csv
import pandas as pd

import matplotlib.pyplot as plt
from matplotlib import cm
from mpl_toolkits.axes_grid1.inset_locator import inset_axes
import matplotlib.gridspec as grsp
import seaborn as sns

from gpytGPE.utils.design import read_labels
from Historia.history import hm

from GSA_library.gsa_parameters_ranking import correct_S
from GSA_library.file_utils import read_ionic_output

# sns.reset_orig()

def plot_pairwise_waves(XL, 
                        colors, 
                        xlabels, 
                        wnames, 
                        figname=None):
    """
    Plot a vector XL of overlapping high-dimensional datasets by means of pairwise components plotting.
    Args:
            - XL: list of L matrices with dimensions n*m_i, for i=1,...,L
            - colors: list of L colors
            - xlabels: list of n strings representing the name of X_is datasets' common features.
    """
    handles, labels = (0, 0)
    L = len(XL)
    in_dim = XL[0].shape[1]
    height = 9.36111
    width = 5.91667
    fig, axes = plt.subplots(
        nrows=in_dim,
        ncols=in_dim,
        sharex="col",
        sharey="row",
        figsize=(2 * width, 2 * height / 2),
    )
    for t, ax in enumerate(axes.flatten()):
        i = t % in_dim
        j = t // in_dim
        if j >= i:
            sns.scatterplot(
                ax=ax,
                x=XL[0][:, i],
                y=XL[0][:, j],
                color=colors[0],
                edgecolor=colors[0],
                label=wnames[0],
                s=5.0
            )
            for k in range(1, L):
                sns.scatterplot(
                    ax=ax,
                    x=XL[k][:, i],
                    y=XL[k][:, j],
                    color=colors[k],
                    edgecolor=colors[k],
                    label=wnames[k],
                    s=5.0,
                )
                handles, labels = ax.get_legend_handles_labels()
                ax.get_legend().remove()
        else:
            ax.set_axis_off()

        if xlabels is not None:
            if i == 0:
                ax.set_ylabel(xlabels[j])
                ax.set_yticks([])
            if j == in_dim - 1:
                ax.set_xlabel(xlabels[i])
                ax.set_xticks([])
            if i == in_dim - 1 and j == 0:
                ax.legend(handles, labels, loc="center")
    # plt.tight_layout()

    # plt.rc('text', usetex=True)
    # plt.rc('font', family='serif')

    if figname:
        # plt.tight_layout()
        plt.savefig(figname,dpi=100)
    else:
        plt.show()
    return

def plot_waves_paramSpace(waves_folder,
                          waves,
                          xlabels,
                          figname,
                          idx_param=None,
                          Ncolors=None):

    """
    Plots the non-implausible area of a series of history matching waves.

    Args: 
        - waves_folder: folder where all wave folders are
        - waves: list of waves to plot [1,2,3,4,...]
        - xlabels: parameter names
        - figname: figure to save
        - idx_param: numpy array of indices of parameters to plot if you don't want to plot them all
        - Ncolors: how many colours you want to use
    """

    plt.close('all')

    if idx_param is None:
        idx_param = np.arange(len(xlabels))
    else:
        idx_param = np.array(idx_param,dtype=int)
        if idx_param.shape[0]!=len(xlabels):
            raise Exception("The labels you provided and the number of parameters you requested do not match.")

    if Ncolors is None:
        Ncolors = len(waves)+1

    evenly_spaced_interval = np.linspace(0, 1, Ncolors)
    colors = [cm.coolwarm(x) for x in evenly_spaced_interval]

    W = hm.Wave()
    W.load(waves_folder+"/wave"+str(waves[0])+"/wave_"+str(waves[0])) 

    X_init = np.concatenate((W.NIMP,W.IMP))
    input_dim = X_init.shape[1]

    XL = [X_init[:,idx_param]]
    wnames = ['initial']
    for w in waves:

        W = hm.Wave()
        W.load(waves_folder+"/wave"+str(w)+"/wave_"+str(w)) 

        NIMP = W.NIMP[:,idx_param]
        # X_wave = np.loadtxt(waves_folder+"/wave"+str(w)+"/X_simul_"+str(w)+".txt", dtype=np.float64)
        # XL.append(X_wave)
        XL.append(NIMP)

        wnames.append('wave '+str(w))

    plot_pairwise_waves(XL,colors,xlabels,wnames,figname=figname)

def compute_ED_idx_dpdt(time,pressure):

    dp = np.diff(pressure)/np.diff(time)
    dpdtmax = np.max(dp)

    ind_ED = np.where(dp>dpdtmax*0.1)[0][0]

    return ind_ED

def comput_ED_idx_IVC(time,volume,pressure):

    time_pmax = time[0]+np.where(pressure==np.max(pressure))[0][0]
    
    dv = np.gradient(volume)

    ind_IVC_ = np.intersect1d(np.where(np.abs(dv)<=0.01)[0],np.where(time<=time_pmax-10.)[0])
    jump = np.where(np.gradient(ind_IVC_)>1)[0]

    if len(jump) == 0:
        ind_IVC = ind_IVC_
    else:
        ind_IVC = ind_IVC_[jump[-1]:-1]

    ind_ED = ind_IVC[0]

    return ind_ED

def compute_end_IVR_idx(time,volume,pressure):

    time_pmax = time[0]+np.where(pressure==np.max(pressure))[0][0]
    
    dv = np.gradient(volume)

    ind_IVR_ = np.intersect1d(np.where(np.abs(dv)<=0.01)[0],np.where(time>=time_pmax+10.)[0])

    if np.size(ind_IVR_)==0:
        ind_end_IVR = np.argmin(pressure)
    else:
        jump = np.where(np.gradient(ind_IVR_)>1)[0] 
        if len(jump) == 0 or jump[-1]<=1:
            ind_IVR = ind_IVR_
        else:
            ind_IVR = ind_IVR_[0:jump[0]]

        ind_end_IVR = ind_IVR[-1]

    return ind_end_IVR

def la_timing(time,volume):

    ESVla = np.array([np.argmin(volume),np.min(volume)])

    end_awave = np.argmin(volume)
    maxV = np.array([end_awave+np.argmax(volume[end_awave:]),np.max(volume[end_awave:])])

    return ESVla,maxV

def plot_waves_pv_fch(sim_folder_list,
                      mask_file_list,
                      figname,
                      BCL,
                      basename="cycle_",
                      target_pressure=None,
                      target_volume=None,
                      figsize=(10,10),
                      fontsize=None):

    """
    Plots the simulated calcium or tension transients at each wave,
    used to refine the emulators.

    Args: 
        - sim_folder_list: list of where the simulations are for all waves
        - mask_file_list: list of mask files to tell the code which simulations did not crash
        - figname: figure to save
        - BCL: basic cycle length of the simulations
        - figsize: size of output figure
        - fontsize: size of the font on the figure
    """

    if fontsize is not None:
        plt.rcParams.update({'font.size': fontsize})

    plt.close('all')

    if len(sim_folder_list)!=len(mask_file_list):
        raise Exception("Length of simulation folders and mask files do not match.")

    idx_ok = []
    for mask_file in mask_file_list:
        mask = np.loadtxt(mask_file,dtype=int)
        idx_ok.append(np.where(mask==1)[0])

    evenly_spaced_interval = np.linspace(0, 1, len(sim_folder_list))
    colors = [cm.coolwarm(x) for x in evenly_spaced_interval]

    ax = plt.figure(figsize=figsize, constrained_layout=True).subplots(1, 2)

    # plot target trace
    if target_volume is not None and target_volume is not None:

        clinical_pressure = np.loadtxt(target_pressure,dtype=float)
        clinical_volume = np.loadtxt(target_volume,dtype=float)

        clinical_time = np.linspace(0., 1., num=clinical_pressure.shape[0], endpoint=True)
        clinical_time *= BCL

        ax[0].plot(clinical_volume,clinical_pressure,'--',
                   color='black',
                   linewidth=2.0)

        ax[1].plot(clinical_time,clinical_pressure,'--',
                   color='black',
                   linewidth=2.0)

    for j,sim_folder in enumerate(sim_folder_list):
        for i in idx_ok[j]:
            lv = read_csv(sim_folder+'/'+basename+str(i)+'/cav.LV.csv', delimiter=",", skipinitialspace=True,
                                        header=0, comment='#')      
            time = np.array(lv['Time'])
            start = time[-1]-BCL
            last_beat = np.where(time>=start)[0]
            time = np.array(lv['Time'][last_beat])  

            volume = np.array(lv['Volume'][last_beat])
            pressure = np.array(lv['Pressure'][last_beat])
            
            ax[0].plot(volume,pressure,color=colors[j],zorder=0,linewidth=1.0)
            ax[0].set_xlabel('Volume [mL]')
            ax[0].set_ylabel('Pressure [mmHg]')

            ind_ED = compute_ED_idx_dpdt(time,pressure)
            pressure = np.concatenate((pressure[ind_ED:],pressure[0:ind_ED]))
            
            ax[1].plot(time-time[0],pressure,color=colors[j],zorder=0,linewidth=1.0)
            ax[1].set_xlabel('Time [ms]')
            ax[1].set_ylabel('Pressure [mmHg]')
            ax[1].set_xlim([0,BCL])

    plt.savefig(figname,dpi=200)

def plot_waves_pv_fch_fitting_v0(sim_folder_list,
                                 mask_file_list,
                                 figname,
                                 BCL,
                                 matchfolder,
                                 basename="cycle_",
                                 pressure_trace=None,
                                 volume_trace=None,
                                 std_factor=3.0,
                                 figsize=(20,10),
                                 fontsize=None):

    print("ASSUMING A PRE-DEFINED ORDER FOR THE FEATURES.")

    exp_mean = np.loadtxt(matchfolder+"/exp_mean.txt",dtype=float)
    exp_std = np.loadtxt(matchfolder+"/exp_std.txt",dtype=float)

    idx_ok = []
    for mask_file in mask_file_list:
        mask = np.loadtxt(mask_file,dtype=int)
        idx_ok.append(np.where(mask==1)[0])

    if pressure_trace is not None:
        clinical_pressure = np.loadtxt(pressure_trace,dtype=float)
        clinical_time = np.linspace(0,BCL,num=clinical_pressure.shape[0], endpoint=True)
        clinical_time_pMax = clinical_time[np.argmax(clinical_pressure)]

    if volume_trace is not None:
        clinical_volume = np.loadtxt(volume_trace,dtype=float)

    fig, ax = plt.subplots(len(mask_file_list), 5,figsize=figsize)

    for i in range(5):
        for j in range(len(sim_folder_list)):
            if j==0:
                ax[j,i] = plt.subplot(len(mask_file_list), 5, i+1+5*j,  sharey=ax[1,i], sharex=ax[1,i])
            else:
                ax[j,i] = plt.subplot(len(mask_file_list), 5, i+1+5*j,  sharey=ax[0,i], sharex=ax[0,i])


        # ax[0,i] = plt.subplot(len(mask_file_list), 5, i+1,  sharey=ax[1,i], sharex=ax[1,i])
        # ax[1,i] = plt.subplot(len(mask_file_list), 5, i+6,  sharey=ax[0,i], sharex=ax[0,i])
        # ax[2,i] = plt.subplot(len(mask_file_list), 5, i+11, sharey=ax[0,i], sharex=ax[0,i])

    for j,sim_folder in enumerate(sim_folder_list):

        pMax = np.zeros((len(idx_ok[j]),2),dtype=float)
        EDV = np.zeros((len(idx_ok[j]),2),dtype=float)
        EDP = np.zeros((len(idx_ok[j]),2),dtype=float)
        ESV = np.zeros((len(idx_ok[j]),2),dtype=float)
        dpdtMax = np.zeros((len(idx_ok[j]),2),dtype=float)
        dpdtMin = np.zeros((len(idx_ok[j]),2),dtype=float)
        EDVrv = np.zeros((len(idx_ok[j]),2),dtype=float)
        ESVrv = np.zeros((len(idx_ok[j]),2),dtype=float)
        ESVla = np.zeros((len(idx_ok[j]),2),dtype=float)
        maxVla = np.zeros((len(idx_ok[j]),2),dtype=float)
        EDVla = np.zeros((len(idx_ok[j]),2),dtype=float)

        for ii,i in enumerate(idx_ok[j]):
            lv = read_csv(sim_folder+'/'+basename+str(i)+'/cav.LV.csv', delimiter=",", skipinitialspace=True,
                                        header=0, comment='#')      
            time = np.array(lv['Time'])
            start = time[-1]-BCL
            last_beat = np.where(time>=start)[0]
            time = np.array(lv['Time'][last_beat])  

            volume = np.array(lv['Volume'][last_beat])
            pressure = np.array(lv['Pressure'][last_beat])
            
            if pressure_trace is not None:
                ax[j,0].plot(clinical_volume,clinical_pressure,color='red',zorder=0,linewidth=1.0)

            ax[j,0].plot(volume,pressure,color='black',zorder=0,linewidth=1.0)
            ax[j,0].set_xlabel('Volume [mL]')
            ax[j,0].set_ylabel('Pressure [mmHg]')

            dpdt = np.zeros((pressure.shape[0],),dtype=float)
            dpdt[1:] = np.diff(pressure)/np.diff(time)*1000.

            # ind_ED_lv = compute_ED_idx_dpdt(time,pressure)
            ind_ED_lv = comput_ED_idx_IVC(time,volume,pressure)
            ind_end_IVR = compute_end_IVR_idx(time,volume,pressure)

            EDV[ii,:] = np.array([volume[ind_ED_lv],pressure[ind_ED_lv]])
            EDP[ii,:] = np.array([ind_ED_lv,pressure[ind_ED_lv]])
            ESV[ii,:] = np.array([volume[ind_end_IVR],pressure[ind_end_IVR]])

            pMax[ii,:] = np.array([np.argmax(pressure),pressure[np.argmax(pressure)]])
            dpdtMax[ii,:] = np.array([np.argmax(dpdt),dpdt[np.argmax(dpdt)]])
            dpdtMin[ii,:] = np.array([np.argmin(dpdt),dpdt[np.argmin(dpdt)]])

            if pressure_trace is not None:
                time_0 = time-time[0]
                time_pMax = time_0[np.argmax(pressure)]

                idx_shift = int(time_pMax-clinical_time_pMax)

                pressure_shift = np.concatenate((pressure[idx_shift:],pressure[0:idx_shift]))

                pressure = pressure_shift
                EDP[ii,0] = EDP[ii,0]-idx_shift
                pMax[ii,0] = pMax[ii,0]-idx_shift

            if pressure_trace is not None:
                ax[j,1].plot(clinical_time,clinical_pressure,color='red',zorder=0,linewidth=1.0)
                
            ax[j,1].plot(time-time[0],pressure,color='black',zorder=0,linewidth=1.0)
            ax[j,1].set_xlabel('Time [ms]')
            ax[j,1].set_ylabel('Pressure [mmHg]')
            ax[j,1].set_xlim([0,BCL])

            ax[j,2].plot(time-time[0],dpdt,color='black',zorder=0,linewidth=1.0)
            ax[j,2].set_xlabel('Time [ms]')
            ax[j,2].set_ylabel('dp/dt [mmHg/s]')
            ax[j,2].set_xlim([0,BCL])

            # -------------------------------------------------------------------
            rv = read_csv(sim_folder+'/'+basename+str(i)+'/cav.RV.csv', delimiter=",", skipinitialspace=True,
                                        header=0, comment='#')      
            time = np.array(rv['Time'])
            start = time[-1]-BCL
            last_beat = np.where(time>=start)[0]
            time = np.array(rv['Time'][last_beat])  

            volume = np.array(rv['Volume'][last_beat])
            pressure = np.array(rv['Pressure'][last_beat])

            # ind_ED = compute_ED_idx_dpdt(time,pressure)
            ind_ED = comput_ED_idx_IVC(time,volume,pressure)
            ind_end_IVR = compute_end_IVR_idx(time,volume,pressure)

            EDVrv[ii,:] = np.array([volume[ind_ED],pressure[ind_ED]])
            ESVrv[ii,:] = np.array([volume[ind_end_IVR],pressure[ind_end_IVR]])

            ax[j,3].plot(volume,pressure,color='black',zorder=0,linewidth=1.0)
            ax[j,3].set_xlabel('Volume [mL]')
            ax[j,3].set_ylabel('Pressure [mmHg]')

            # -------------------------------------------------------------------
            la = read_csv(sim_folder+'/'+basename+str(i)+'/cav.LA.csv', delimiter=",", skipinitialspace=True,
                                        header=0, comment='#')      
            time = np.array(la['Time'])
            start = time[-1]-BCL
            last_beat = np.where(time>=start)[0]
            time = np.array(la['Time'][last_beat])  

            volume = np.array(la['Volume'][last_beat])

            ESVla[ii,:],maxVla[ii,:] = la_timing(time,volume)

            EDVla[ii,:] = np.array([0,volume[0]])

            ax[j,4].plot(time-time[0],volume,color='black',zorder=0,linewidth=1.0)
            ax[j,4].set_xlabel('Time [ms]')
            ax[j,4].set_ylabel('Volume [mL]')
            ax[j,4].set_xlim([-20,BCL+20])
    
        ax[j,0].scatter(EDV[:,0],EDV[:,1],s=5.0,color='deepskyblue', zorder=10)
        ax[j,0].scatter(ESV[:,0],ESV[:,1],s=5.0,color='orange', zorder=10)

        ax[j,1].scatter(EDP[:,0],EDP[:,1],s=5.0,color='deepskyblue', zorder=10)
        ax[j,1].scatter(pMax[:,0],pMax[:,1],s=5.0,color='orange', zorder=10)

        ax[j,2].scatter(dpdtMax[:,0],dpdtMax[:,1],s=5.0,color='deepskyblue', zorder=10)
        ax[j,2].scatter(dpdtMin[:,0],dpdtMin[:,1],s=5.0,color='orange', zorder=10)

        ax[j,3].scatter(EDVrv[:,0],EDVrv[:,1],s=5.0,color='deepskyblue', zorder=10)
        ax[j,3].scatter(ESVrv[:,0],ESVrv[:,1],s=5.0,color='orange', zorder=10)

        ax[j,4].scatter(EDVla[:,0],EDVla[:,1],s=5.0,color='deepskyblue', zorder=10)
        ax[j,4].scatter(ESVla[:,0],ESVla[:,1],s=5.0,color='orange', zorder=10)
        ax[j,4].scatter(maxVla[:,0],maxVla[:,1],s=5.0,color='magenta', zorder=10)

    for j in range(len(sim_folder_list)):

        ymin,ymax = ax[j,0].get_ylim()
        ax[j,0].vlines(x=exp_mean[0], ymin=ymin, ymax=ymax, linewidth=3, 
                       color='deepskyblue',zorder=20)
        ax[j,0].vlines(x=exp_mean[2], ymin=ymin, ymax=ymax, linewidth=3, 
                       color='orange',zorder=20)
        ax[j,0].set_ylim([ymin,ymax])

        xmin,xmax = ax[j,1].get_xlim()
        ax[j,1].hlines(y=exp_mean[1], xmin=xmin, xmax=xmax, linewidth=3, 
                       color='deepskyblue',zorder=20)
        ax[j,1].hlines(y=exp_mean[3], xmin=xmin, xmax=xmax, linewidth=3, 
                       color='orange',zorder=20)

        ax[j,2].hlines(y=exp_mean[4], xmin=xmin, xmax=xmax, linewidth=3, 
                       color='deepskyblue',zorder=20)
        ax[j,2].hlines(y=exp_mean[5], xmin=xmin, xmax=xmax, linewidth=3, 
                       color='orange',zorder=20)

        ymin,ymax = ax[j,3].get_ylim()
        ax[j,3].vlines(x=exp_mean[6], ymin=ymin, ymax=ymax, linewidth=3, 
                       color='deepskyblue',zorder=20)
        ax[j,3].vlines(x=exp_mean[8], ymin=ymin, ymax=ymax, linewidth=3, 
                       color='orange',zorder=20)
        ax[j,3].set_ylim([ymin,ymax])

        xmin,xmax = ax[j,1].get_xlim()
        ax[j,4].hlines(y=exp_mean[12], xmin=xmin, xmax=xmax, linewidth=3, 
                       color='deepskyblue',zorder=20)
        ax[j,4].hlines(y=exp_mean[13], xmin=xmin, xmax=xmax, linewidth=3, 
                       color='orange',zorder=20)
        ax[j,4].hlines(y=exp_mean[14], xmin=xmin, xmax=xmax, linewidth=3, 
                       color='magenta',zorder=20)

    plt.tight_layout()
    plt.savefig(figname,dpi=200)

def plot_waves_dynamics(waves_folder,
                        N,
                        waves,
                        initialfolder,
                        figname,
                        variable='Ca_i',
                        target='',
                        Ncolors=None,
                        figsize=(9,3),
                        fontsize=None):

    """
    Plots the simulated calcium or tension transients at each wave,
    used to refine the emulators.

    Args: 
        - waves_folder: folder where all wave folders are
        - N: list of number of simulations in each dataset
        - waves: list of waves to plot [1,2,3,4,...]
        - initialfolder: where to find the initial set of simulations. initialfolder/sims/
        - figname: figure to save
        - variable: Ca_i or Tension
        - Ncolors: how many colours you want to use
        - figsize: size of output figure
        - fontsize: size of the font on the figure
    """

    if fontsize is not None:
        plt.rcParams.update({'font.size': fontsize})

    plt.close('all')

    if variable=='Ca_i':
        ylabel = '$[Ca^{2+}]_i$'
        units = '[um]'
    elif variable=='Tension':
        ylabel = 'Tension'
        units = '[kPa]'
    else:
        raise Exception("Wrong variable. Select Ca_i or Tension")

    if Ncolors is None:
        Ncolors = len(waves)+1

    evenly_spaced_interval = np.linspace(0, 1, Ncolors)
    colors = [cm.coolwarm(x) for x in evenly_spaced_interval]

    fig,ax = plt.subplots(1,1,figsize=figsize, constrained_layout=True)
    for i in range(len(waves)+1):
        for n in range(N[i]):
            if i == 0:
                folder = initialfolder + "/sims/"
            else:
                folder = waves_folder+"/wave"+str(waves[i-1])+"/sims/"
            data = read_ionic_output(folder+'/'+str(n)+'/'+variable+'.dat')
            t = np.arange(0,data.shape[0])

            ax.set_xlim(0,data.shape[0])
            ax.plot(t,data,color=colors[i])
            ax.set_xlabel('Time [ms]')
            ax.set_ylabel(ylabel+' '+units)

    # plot target trace
    if target != '':
        target_trace = read_ionic_output(target)
        ax.plot(t,target_trace,'--',
                    color='black',
                    linewidth=2.0
                    )

    plt.savefig(figname,dpi=100)

def plot_Y_files_dataset(Y_files_list,
                       ylabels,
                       datafolder,
                       figname,
                       figsize=(10,5),
                       idx_features=None,
                       std_factor=3.0,
                       above_0=False):

    """
    Plots the outputs for the simulations used to refine the GPEs
    at each history matching wave, compared to the dataset.

    Args: 
        - Y_files_list: list of output files to plot
        - ylabels: output labels
        - datafolder: folder containing the mean and standard deviation for the data
        - figname: figure to save
        - figsize: size of output figure
        - idx_features: list of features to plot
        - std_factor: how many experimental standard deviations you want to plot
        - above_0: limit the y-scale to be above 0 (True/False)
    """

    m = np.loadtxt(datafolder+'/exp_mean.txt')
    s = np.loadtxt(datafolder+'/exp_std.txt')

    if idx_features is None:
        idx_features_list = range(m.shape[0])
    else:
        idx_features_list = list(np.loadtxt(idx_features,dtype=int))

    evenly_spaced_interval = np.linspace(0, 1, len(Y_files_list))
    colors = [cm.coolwarm(x) for x in evenly_spaced_interval]

    N_features = len(idx_features_list)

    fig,ax = plt.subplots(1,N_features,figsize=figsize, constrained_layout=True)

    waves = ["wave"+str(i) for i in range(len(Y_files_list))]

    for n in range(N_features):

        for i,w in enumerate(waves):
            Y = np.loadtxt(Y_files_list[i], dtype=float)
            N = Y.shape[0]
            ax[n].plot(np.ones((N,1))+i,Y[:,idx_features_list[n]],linestyle='',
                                marker='o',
                                markersize=4.0,
                                mfc=colors[i],
                                color=colors[i])

        ax[n].errorbar([len(waves)+1], m[idx_features_list[n]], s[idx_features_list[n]]*std_factor, marker='o',color='black', mfc='black',
            markersize=5.0)

        ax[n].set_xlim([0,len(waves)+2])
        ax[n].set_xticks(range(1,len(waves)+2))
        ax[n].set_xticklabels(waves+['dataset'],rotation=90)

        ylim = ax[n].get_ylim()
        if above_0 and (ylim[0]<0):
            ax[n].set_ylim(bottom=0) 

        ax[n].set_ylabel(ylabels[idx_features_list[n]])

    plt.rc('text', usetex=True)
    plt.rc('font', family='serif')

    plt.savefig(figname,dpi=100)#

def plot_waves_dataset(waves_folder,
                       waves,
                       initialfolder,
                       ylabels,
                       datafolder,
                       figname,
                       figsize=(10,5),
                       idx_features=None,
                       std_factor=3.0,
                       above_0=False):

    """
    Plots the outputs for the simulations used to refine the GPEs
    at each history matching wave, compared to the dataset.

    Args: 
        - waves_folder: folder where all wave folders are
        - waves: list of waves to plot [1,2,3,4,...]
        - initialfolder: where to find the first outputs. initialfolder/Y.txt
        - ylabels: output labels
        - datafolder: folder containing the mean and standard deviation for the data
        - figname: figure to save
        - figsize: size of output figure
        - idx_features: list of features to plot
        - std_factor: how many experimental standard deviations you want to plot
        - above_0: limit the y-scale to be above 0 (True/False)
    """

    m = np.loadtxt(datafolder+'/exp_mean.txt')
    s = np.loadtxt(datafolder+'/exp_std.txt')

    if idx_features is None:
        idx_features_list = range(m.shape[0])
    else:
        idx_features_list = list(np.loadtxt(idx_features,dtype=int))

    evenly_spaced_interval = np.linspace(0, 1, len(waves)+1)
    colors = [cm.coolwarm(x) for x in evenly_spaced_interval]

    N_features = len(idx_features_list)

    fig,ax = plt.subplots(1,N_features,figsize=figsize, constrained_layout=True)

    for n in range(N_features):

        Y = np.loadtxt(initialfolder+'/Y.txt')
        N = Y.shape[0]

        ax[n].plot(np.ones((N,1)),Y[:,idx_features_list[n]],linestyle='',
                                marker='o',
                                markersize=2.0,
                                mfc=colors[0],
                                color=colors[0])
        labels = ['Initial dataset']
        for i,w in enumerate(waves):
            Y = np.loadtxt(waves_folder+"/wave"+str(w)+"/Y_simul_"+str(w)+".txt", dtype=float)
            N = Y.shape[0]
            ax[n].plot(np.ones((N,1))+i+1,Y[:,idx_features_list[n]],linestyle='',
                                marker='o',
                                markersize=2.0,
                                mfc=colors[i+1],
                                color=colors[i+1])
            labels.append('wave '+str(w))

        labels.append('Dataset')

        ax[n].errorbar([len(waves)+2], m[idx_features_list[n]], s[idx_features_list[n]]*std_factor, marker='o',color='black', mfc='black',
            markersize=5.0)

        ax[n].set_xlim([0,len(waves)+3])
        ax[n].set_xticks(range(1,len(waves)+3))
        ax[n].set_xticklabels(labels,rotation=90)

        ylim = ax[n].get_ylim()
        if above_0 and (ylim[0]<0):
            ax[n].set_ylim(bottom=0) 

        ax[n].set_ylabel(ylabels[idx_features_list[n]])

    plt.rc('text', usetex=True)
    plt.rc('font', family='serif')

    plt.savefig(figname,dpi=100)#

def plot_R2_HM(loadpath,
               dataFolder,
               figname=None):

    """
    Plots the R2 scores for the GPEs for a series of waves.

    Args: 
        - loadpath: list of GPE paths
        - dataFolder: folder containing the ylabels.txt and features_idx_list_hm.txt
        - figname: optional figure name
    """

    files_to_check = [dataFolder+"/features_idx_list_hm.txt",
                      dataFolder+"/ylabels.txt"]

    for f in files_to_check:
        if not os.path.exists(f):
            raise Exception("Cannot find file "+f+".")

    evenly_spaced_interval = np.linspace(0, 1, len(loadpath))
    colors = [cm.coolwarm(x) for x in evenly_spaced_interval]

    data = {}
    idx_features = np.loadtxt(dataFolder+"/features_idx_list_hm.txt",dtype=int)
    ylabels = read_labels(dataFolder+"/ylabels.txt")

    fig, axes = plt.subplots(
        nrows=1,
        ncols=len(idx_features),
        figsize=(15,3),
    )
    for i,idx in enumerate(idx_features):
        data[str(idx)] = []
        for j,path in enumerate(loadpath):
            R2_score = np.loadtxt(path+"/"+str(idx)+"/R2Score_cv.txt")
            data[str(idx)].append(R2_score)
            axes[i].plot(np.ones(len(R2_score),)*j,R2_score,'o',color=colors[j])
        if i==0:
            axes[i].set_ylabel('R2 score')
        axes[i].set_title(ylabels[i])
    if figname is None:
        plt.show()
    else:
        plt.savefig(figname)

def plot_wave_X_subset(wave_file, 
                       idx_param,
                       xlabels, 
                       display="impl", 
                       filename="./wave_impl"):

    """
    Plots non-implausibility or uncertainty map for a subset of parameters

    Args: 
        - wave_file: wave to use
        - idx_param: list of the parameters to plot
        - xlabels: list of parameters names
        - display: impl or var. Display implausibility of uncertainty
        - filename: output figure
    """

    W = hm.Wave()
    W.load(wave_file)

    X = W.reconstruct_tests()
    X_plot = X[:,idx_param]

    W.input_dim = idx_param.shape[0]

    if len(xlabels) != idx_param.shape[0]:
        raise Exception('xlabels and parameter indices do not match.')

    if display == "impl":
        C = W.I
        cmap = "jet"
        vmin = 1.0
        vmax = W.cutoff
        cbar_label = "Implausibility measure"

    elif display == "var":
        C = W.PV
        cmap = "bone_r"
        vmin = np.max(
            [
                np.percentile(W.PV, 25) - 1.5 * iqr(W.PV),
                W.PV.min(),
            ]
        )
        vmax = np.min(
            [
                np.percentile(W.PV, 75) + 1.5 * iqr(W.PV),
                W.PV.max(),
            ]
        )
        cbar_label = "GPE variance / EXP. variance"

    else:
        raise ValueError(
            "Not a valid display option! Can only display implausibilty maps ('impl') or proportion-of-exp.variance maps ('var')."
        )

    height = 9.36111
    width = 5.91667
    fig = plt.figure(figsize=(3 * width, 3 * height / 3))
    gs = grsp.GridSpec(
        W.input_dim - 1,
        W.input_dim,
        width_ratios=(W.input_dim - 1) * [1] + [0.1],
    )

    for k in range(W.input_dim * W.input_dim):
        i = k % W.input_dim
        j = k // W.input_dim

        if i > j:
            axis = fig.add_subplot(gs[i - 1, j])
            axis.set_facecolor("xkcd:light grey")

            im = axis.hexbin(
                X[:, j],
                X[:, i],
                C=C,
                reduce_C_function=np.min,
                gridsize=20,
                cmap=cmap,
                vmin=vmin,
                vmax=vmax,
            )

            axis.set_xlim([W.Itrain[j, 0], W.Itrain[j, 1]])
            axis.set_ylim([W.Itrain[i, 0], W.Itrain[i, 1]])

            if i == W.input_dim - 1:
                axis.set_xlabel(xlabels[j], fontsize=10)
                axis.set_xticklabels([])
            else:
                axis.set_xticklabels([])
            if j == 0:
                axis.set_ylabel(xlabels[i], fontsize=10)
                axis.set_yticklabels([])
            else:
                axis.set_yticklabels([])

    cbar_axis = fig.add_subplot(gs[:, W.input_dim - 1])
    cbar = fig.colorbar(im, cax=cbar_axis)
    cbar.set_label(cbar_label, size=12)
    fig.tight_layout()
    plt.savefig(filename + ".png", bbox_inches="tight", dpi=300)

def plot_wave_subset(wave_file, 
                     xlabels, 
                     idx_param=None,
                     display="impl", 
                     filename="./wave_impl"):

##################### NOT TESTED

    W = hm.Wave()
    W.load(wave_file)

    X = W.reconstruct_tests()
    X = X[:,idx_param]
    W.input_dim = idx_param.shape[0]
    W.Itrain = W.Itrain[idx_param,:]

    if idx_param is None:
        idx_param = np.arange(len(xlabels))
    else:
        if idx_param.shape[0]!=len(xlabels):
            raise Exception("The labels you provided and the number of parameters you requested do not match.")

    if display == "impl":
        C = W.I
        cmap = "jet"
        vmin = 1.0
        vmax = W.cutoff
        cbar_label = "Implausibility measure"

    elif display == "var":
        C = W.PV
        cmap = "bone_r"
        vmin = np.max(
            [
                np.percentile(W.PV, 25) - 1.5 * iqr(W.PV),
                W.PV.min(),
            ]
        )
        vmax = np.min(
            [
                np.percentile(W.PV, 75) + 1.5 * iqr(W.PV),
                W.PV.max(),
            ]
        )
        cbar_label = "GPE variance / EXP. variance"

    else:
        raise ValueError(
            "Not a valid display option! Can only display implausibilty maps ('impl') or proportion-of-exp.variance maps ('var')."
        )

    height = 9.36111
    width = 5.91667
    fig = plt.figure(figsize=(3 * width, 3 * height / 3))
    gs = grsp.GridSpec(
        W.input_dim - 1,
        W.input_dim,
        width_ratios=(W.input_dim - 1) * [1] + [0.1],
    )

    for k in range(W.input_dim * W.input_dim):
        i = k % W.input_dim
        j = k // W.input_dim

        if i > j:
            axis = fig.add_subplot(gs[i - 1, j])
            axis.set_facecolor("xkcd:light grey")

            im = axis.hexbin(
                X[:, j],
                X[:, i],
                C=C,
                reduce_C_function=np.min,
                gridsize=20,
                cmap=cmap,
                vmin=vmin,
                vmax=vmax,
            )

            axis.set_xlim([W.Itrain[j, 0], W.Itrain[j, 1]])
            axis.set_ylim([W.Itrain[i, 0], W.Itrain[i, 1]])

            if i == W.input_dim - 1:
                axis.set_xlabel(xlabels[j], fontsize=12)
            else:
                axis.set_xticklabels([])
            if j == 0:
                axis.set_ylabel(xlabels[i], fontsize=12)
            else:
                axis.set_yticklabels([])

    cbar_axis = fig.add_subplot(gs[:, W.input_dim - 1])
    cbar = fig.colorbar(im, cax=cbar_axis)
    cbar.set_label(cbar_label, size=12)
    fig.tight_layout()
    plt.savefig(filename + ".png", bbox_inches="tight", dpi=300)
