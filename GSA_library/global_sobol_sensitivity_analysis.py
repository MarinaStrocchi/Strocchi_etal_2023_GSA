import os
import sys

import numpy as np
from scipy.special import binom
from scipy.stats import norm

import random
from itertools import combinations

import torch
from SALib.analyze import sobol
from SALib.sample import saltelli

from gpytGPE.gpe import GPEmul
from gpytGPE.utils.design import get_minmax, read_labels
from gpytGPE.utils.plotting import gsa_box, gsa_donut, gsa_network
from gpytGPE.utils.design import lhd

from Historia.history import hm

from GSA_library.plotting import plot_dataset

from GSA_library import sobol_analyze_NIMP
from GSA_library.saltelli_pick_sampling import sample_NIMP

EMUL_TYPE = "full"  # possible choices are: "full", "best"
N = 1000
N_BOOTSTRAP = 100
N_DRAWS = 1000

# N = 100
# N_BOOTSTRAP = 10
# N_DRAWS = 10

SEED = 8
THRE = 0.01
WATCH_METRIC = "R2Score"
HIGHEST_IS_BEST = True


def global_sobol_sensitivity_analysis(loadpath,
                                      idx_feature,
                                      savepath,
                                      sample_method='sobol',
                                      calc_second_order=True,
                                      uncertainty=True,
                                      n_factor=1):

    """
    Performs a Sobol GSA given trained GPEs and a list of features.

    Args:
        - loadpath: datafolder containing data used to train GPE
        - idx_feature: which feature to perform the GSA for
        - savepath: where to save the GSA
        - sample_method: 'sobol' or 'lhd' - to test if the GSA makes a difference
                         'sobol' by default
        - calc_second_order: True if you want to compute second order effects too
        - uncertainty: if True, the posterior distribution of the GPEs is 
                       is sampled N=1000 times and the sensitivity indices
                       are computed for each sample
        - n_factor: increase the number of samples for the GSA by a factor
    """ 

    # ================================================================
    # Making the code reproducible
    # ================================================================
    seed = SEED
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)

    # ================================================================
    # GPE loading
    # ================================================================
    print('Checking folder structure...')
    to_check = [loadpath + "xlabels.txt",savepath]
    for f in to_check:
        if not os.path.exists(f):
            raise Exception('Cannot find '+f)
        else:
            print(f+' found.')

    emul_type = EMUL_TYPE
    metric_name = WATCH_METRIC
    metric_type = HIGHEST_IS_BEST

    print('Using '+emul_type+' GPE for sensitivity analysis...')

    if emul_type == "best":
        if not os.path.exists(savepath + metric_name + "_cv.txt"):
            raise Exception('I need the score file to evaluate which GPE is best.')
        metric_score_list = np.loadtxt(
            savepath + metric_name + "_cv.txt", dtype=float
        )
        if metric_type:
            best_split = np.argmax(metric_score_list)
        else:
            best_split = np.argmin(metric_score_list)
        savepath = savepath + f"{best_split}/"

    if not os.path.exists(savepath + "/X_train.txt"):
        raise Exception('Cannot find '+savepath+'/X_train.txt')
    if not os.path.exists(savepath + "/y_train.txt"):
        raise Exception('Cannot find '+savepath+'/y_train.txt') 
               
    X_train = np.loadtxt(savepath + "/X_train.txt", dtype=float)
    y_train = np.loadtxt(savepath + "/y_train.txt", dtype=float)

    emul = GPEmul.load(X_train, y_train, savepath)

    # ================================================================
    # Estimating Sobol' sensitivity indices
    # ================================================================

    n = N*n_factor
    n_draws = N_DRAWS

    d = X_train.shape[1]
    I = get_minmax(X_train)

    index_i = read_labels(loadpath + "xlabels.txt")
    index_ij = [list(c) for c in combinations(index_i, 2)]

    problem = {"num_vars": d, "names": index_i, "bounds": I}

    if sample_method=='sobol':

        X = saltelli.sample(problem, n, calc_second_order=calc_second_order)

    elif sample_method=='lhd':

        X,sample_weights = sample_NIMP(problem, n, 
                                       calc_second_order=calc_second_order, 
                                       sampling_method='lhd')
    else:
        raise Exception("Please pick sampling_method between sobol or lhd.")

    conf_level = 0.95
    z = norm.ppf(0.5 + conf_level / 2)
    n_bootstrap = N_BOOTSTRAP

    if uncertainty:

        Y = emul.sample(X, n_draws=n_draws)

        ST = np.zeros((0, d), dtype=float)
        S1 = np.zeros((0, d), dtype=float)
        S2 = np.zeros((0, int(binom(d, 2))), dtype=float)   

        ST_std = np.zeros((0, d), dtype=float)
        S1_std = np.zeros((0, d), dtype=float)
        S2_std = np.zeros((0, int(binom(d, 2))), dtype=float)   

        for i in range(n_draws):
            S = sobol.analyze(
                problem,
                Y[i],
                calc_second_order=calc_second_order,
                num_resamples=n_bootstrap,
                conf_level=conf_level,
                print_to_console=False,
                parallel=False,
                n_processors=None,
                seed=seed,
            )   

            T_Si, first_Si, (_, second_Si) = sobol.Si_to_pandas_dict(S) 

            ST = np.vstack((ST, T_Si["ST"].reshape(1, -1)))
            S1 = np.vstack((S1, first_Si["S1"].reshape(1, -1))) 

            if calc_second_order:
                S2 = np.vstack((S2, np.array(second_Si["S2"]).reshape(1, -1)))  

            ST_std = np.vstack((ST_std, T_Si["ST_conf"].reshape(1, -1) / z))
            S1_std = np.vstack((S1_std, first_Si["S1_conf"].reshape(1, -1) / z))    

            if calc_second_order:
                S2_std = np.vstack(
                    (S2_std, np.array(second_Si["S2_conf"]).reshape(1, -1) / z)
                )   

        np.savetxt(savepath + "STi.txt", ST, fmt="%.6f")
        np.savetxt(savepath + "Si.txt", S1, fmt="%.6f")
        np.savetxt(savepath + "Sij.txt", S2, fmt="%.6f")    

        np.savetxt(savepath + "STi_std.txt", ST_std, fmt="%.6f")
        np.savetxt(savepath + "Si_std.txt", S1_std, fmt="%.6f")
        np.savetxt(savepath + "Sij_std.txt", S2_std, fmt="%.6f")

    else:

        Y, std = emul.predict(X)

        S = sobol.analyze(
                        problem,
                        Y,
                        calc_second_order=calc_second_order,
                        num_resamples=n_bootstrap,
                        conf_level=conf_level,
                        print_to_console=False,
                        parallel=False,
                        n_processors=None,
                        seed=seed,
                    )   

        T_Si, first_Si, (_, second_Si) = sobol.Si_to_pandas_dict(S) 

        ST = T_Si["ST"].reshape(1, -1)
        S1 = first_Si["S1"].reshape(1, -1) 

        if calc_second_order:
            S2 = np.array(second_Si["S2"]).reshape(1, -1)
        else:
            S2 = np.zeros((0, int(binom(d, 2))), dtype=float)

        ST_std = T_Si["ST_conf"].reshape(1, -1) / z
        S1_std = first_Si["S1_conf"].reshape(1, -1) / z    

        if calc_second_order:
            S2_std = np.array(second_Si["S2_conf"]).reshape(1, -1) / z
        else:
            S2_std = np.zeros((0, int(binom(d, 2))), dtype=float)
           
        np.savetxt(savepath + "STi.txt", ST, fmt="%.6f")
        np.savetxt(savepath + "Si.txt", S1, fmt="%.6f")
        np.savetxt(savepath + "Sij.txt", S2, fmt="%.6f")    

        np.savetxt(savepath + "STi_std.txt", ST_std, fmt="%.6f")
        np.savetxt(savepath + "Si_std.txt", S1_std, fmt="%.6f")
        np.savetxt(savepath + "Sij_std.txt", S2_std, fmt="%.6f")

def global_sobol_sensitivity_analysis_NIMP(loadpath,
                                           idx_feature,
                                           savepath,
                                           wave_file=None,
                                           wave_gpepath=None,
                                           wave_features_idx=None,
                                           wave_idx_param=None,
                                           sample_method='sobol',
                                           calc_second_order=False,
                                           imp_method='exclude',
                                           X_samples_file=None,
                                           samples_weights_file=None,
                                           n_factor=1):

    """
    Performs a Sobol GSA given trained GPEs and a list of features,
    while using a history matching wave to restrict a subset of 
    the parameters to a region that is not shaped like an hypercube.

    Args:
        - loadpath: datafolder containing data used to train GPE
        - idx_feature: which feature to perform the GSA for
        - savepath: where to save the GSA
        - wave_file: wave used to derive the NIMP 
        - wave_gpepath: path to the trained GPEs you want to  use 
                        to determine whether the points are non-implausible
                        or not
        - wave_features_idx: array containing indices of the features the GPEs
                        were trained for
        - wave_idx_param: array containing the index of which columns of X
                          have been derived with the GPE
        - sample_method: 'sobol' or 'lhd' - to test if the GSA makes a difference
                         'sobol' by default
        - calc_second_order: True if you want to compute second order effects too
        - imp_method: 'exclude' removes the non-implausible samples using the GPEs and
                      a previously run wave of history mathing. 'mean' will set the
                      output values of the implausible region to the mean value
                      of the non-implausible outputs (Jung et al 2022 doi: 10.3390/math10050823)
        - X_samples_file: provide the samples to run the GSA rather than sampling them
        - samples_weights_file: provide a file with binary weigths to specify which samples
                                in the X_samples_file are plausible
        - n_factor: increase the number of samples for the GSA by a factor 
    """

    # ================================================================
    # Making the code reproducible
    # ================================================================
    seed = SEED
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)

    if calc_second_order:
        raise Exception('I have not configured the GSA on a NIMP to account for second order effects yet. Please set to False.')

    # ================================================================
    # GPE loading
    # ================================================================
    print('Checking folder structure...')
    to_check = [loadpath + "xlabels.txt",savepath]
    for f in to_check:
        if not os.path.exists(f):
            raise Exception('Cannot find '+f)
        else:
            print(f+' found.')

    emul_type = EMUL_TYPE
    metric_name = WATCH_METRIC
    metric_type = HIGHEST_IS_BEST

    print('Using '+emul_type+' GPE for sensitivity analysis...')

    if emul_type == "best":
        if not os.path.exists(savepath + metric_name + "_cv.txt"):
            raise Exception('I need the score file to evaluate which GPE is best.')
        metric_score_list = np.loadtxt(
            savepath + metric_name + "_cv.txt", dtype=float
        )
        if metric_type:
            best_split = np.argmax(metric_score_list)
        else:
            best_split = np.argmin(metric_score_list)
        savepath = savepath + f"{best_split}/"

    if not os.path.exists(savepath + "/X_train.txt"):
        raise Exception('Cannot find '+savepath+'/X_train.txt')
    if not os.path.exists(savepath + "/y_train.txt"):
        raise Exception('Cannot find '+savepath+'/y_train.txt') 
               
    X_train = np.loadtxt(savepath + "/X_train.txt", dtype=float)
    y_train = np.loadtxt(savepath + "/y_train.txt", dtype=float)

    emul = GPEmul.load(X_train, y_train, savepath)

    # ================================================================
    # Sampling with the Saltelli method
    # ================================================================
    # sample the space 10 times as we will be getting rid of many points
    n = N*n_factor
    # n = N
    n_draws = N_DRAWS

    d = X_train.shape[1]
    I = get_minmax(X_train)

    index_i = read_labels(loadpath + "xlabels.txt")
    index_ij = [list(c) for c in combinations(index_i, 2)]

    problem = {"num_vars": d, "names": index_i, "bounds": I}

    if sample_method=='lhd':

        if (wave_file is None) or (wave_gpepath is None) or (wave_features_idx is None) or (wave_idx_param is None):
            raise Exception('You need to provide a wave, the path to the GPEs for the wave, the features and the parameters to constrain.')

        X,sample_weights = sample_NIMP(problem, n, 
                        calc_second_order=calc_second_order, 
                        sampling_method='lhd_NIMP',
                        wave_file=wave_file,
                        wave_gpepath=wave_gpepath,
                        wave_features_idx=wave_features_idx,
                        wave_idx_param=wave_idx_param)

    elif sample_method=='sobol':
        if (wave_file is None) or (wave_gpepath is None) or (wave_features_idx is None) or (wave_idx_param is None):
            raise Exception('You need to provide a wave, the path to the GPEs for the wave, the features and the parameters to constrain.')

        X,sample_weights = sample_NIMP(problem, n, 
                        calc_second_order=calc_second_order, 
                        sampling_method='sobol_NIMP',
                        wave_file=wave_file,
                        wave_gpepath=wave_gpepath,
                        wave_features_idx=wave_features_idx,
                        wave_idx_param=wave_idx_param)

    elif sample_method=='saved':
        if X_samples_file is None:
            raise Exception("If you pick sample_method to be saved, you need to provide a file with the samples.")
        else:
            X = np.loadtxt(X_samples_file,dtype=float)
            if samples_weights_file is None:
                sample_weights = np.ones((X.shape[0],),dtype=int)
            else:
                sample_weights = np.loadtxt(samples_weights_file,dtype=int)

    else:
        raise Exception("Please pick sampling_method between sobol, saved or lhd.")

    # ================================================================
    # Estimating Sobol' sensitivity indices
    # ================================================================
    Y = np.zeros((n_draws,X.shape[0]),dtype=float)
    Y_emul = emul.sample(X[np.where(sample_weights==1)[0],:], n_draws=n_draws)
    Y[:,np.where(sample_weights==1)[0]] = Y_emul

    if imp_method=='exclude':
        print('---------------------------------------------------------------------------')
        print('         Excluding A & B crossed samples that fall outside the NIMP        ')
        print('---------------------------------------------------------------------------')

    elif imp_method=='mean':
        print('---------------------------------------------------------------------------')
        print('         Setting IMP crossed samples to the mean value of NIMP output      ')
        print('                             Weights reset to 1                            ')
        print('---------------------------------------------------------------------------')

        imp_idx = np.where(sample_weights==0)[0]
        Y[:,imp_idx] = np.repeat(np.mean(Y_emul,axis=1),
                                 np.ones((N_DRAWS,),dtype=int)*imp_idx.shape[0]).reshape(N_DRAWS,imp_idx.shape[0])
        sample_weights = np.ones((X.shape[0],),dtype=int)
    else:
        raise Exception('Pick imp_method between mean or exclude')  

    conf_level = 0.95
    z = norm.ppf(0.5 + conf_level / 2)
    n_bootstrap = N_BOOTSTRAP

    ST = np.zeros((0, d), dtype=float)
    S1 = np.zeros((0, d), dtype=float)
    S2 = np.zeros((0, int(binom(d, 2))), dtype=float)

    ST_std = np.zeros((0, d), dtype=float)
    S1_std = np.zeros((0, d), dtype=float)
    S2_std = np.zeros((0, int(binom(d, 2))), dtype=float)

    for i in range(n_draws):

        S = sobol_analyze_NIMP.analyze_NIMP(
            problem,
            Y[i],
            calc_second_order=calc_second_order,
            num_resamples=n_bootstrap,
            conf_level=conf_level,
            print_to_console=False,
            parallel=False,
            n_processors=None,
            seed=seed,
            sample_weights=sample_weights,
        )

        T_Si, first_Si, (_, second_Si) = sobol.Si_to_pandas_dict(S)

        ST = np.vstack((ST, T_Si["ST"].reshape(1, -1)))
        S1 = np.vstack((S1, first_Si["S1"].reshape(1, -1)))

        if calc_second_order:
            S2 = np.vstack((S2, np.array(second_Si["S2"]).reshape(1, -1)))

        ST_std = np.vstack((ST_std, T_Si["ST_conf"].reshape(1, -1) / z))
        S1_std = np.vstack((S1_std, first_Si["S1_conf"].reshape(1, -1) / z))

        if calc_second_order:
            S2_std = np.vstack(
                (S2_std, np.array(second_Si["S2_conf"]).reshape(1, -1) / z)
            )

    np.savetxt(savepath + "STi.txt", ST, fmt="%.6f")
    np.savetxt(savepath + "Si.txt", S1, fmt="%.6f")
    np.savetxt(savepath + "Sij.txt", S2, fmt="%.6f")

    np.savetxt(savepath + "STi_std.txt", ST_std, fmt="%.6f")
    np.savetxt(savepath + "Si_std.txt", S1_std, fmt="%.6f")
    np.savetxt(savepath + "Sij_std.txt", S2_std, fmt="%.6f")
