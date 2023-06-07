import os
import sys

import re

import numpy as np

from gpytGPE.utils.design import read_labels
from gpytGPE.utils.plotting import correct_index

from GSA_library.file_utils import write_json

CRITERION = "STi"  # possible choices are: "Si", "STi"
EMUL_TYPE = "full"  # possible choices are: "full", "best"
THRE = 0.0
WATCH_METRIC = "R2Score"
HIGHEST_IS_BEST = True

def correct_S(S,th=0.01):

    """
    Corrects sensitivity matrix assigning 0.0 to values below threshold.

    Args:
        - S: matrix of sensitivity indices to correct
        - th: threshold
    """

    if th<0 or th>1.0:
        raise Exception('Provide threshold between 0 and 1.')

    print('Correcting sensitivity with threshold '+str(th)+'...')

    S[np.where(S<=th)] = 0.0

    return S

def gsa_parameters_ranking(loadpath,loadpath_sobol):

    """
    Ranks parameters using sensitivity indies.

    Args:
        - loadpath: datafolder containing data used for GPE training
        - loadpath_sobol: folder containing GPEs and GSA results
    """

    print('Checking folder structure...')
    to_check = [loadpath + "xlabels.txt",
                loadpath + "ylabels.txt",
                loadpath + "features_idx_list.txt",
                loadpath_sobol]
    for f in to_check:
        if not os.path.exists(f):
            raise Exception('Cannot find file '+f)
        else:
            print(f+' found.')

    index_i = read_labels(loadpath + "xlabels.txt")
    label = read_labels(loadpath + "ylabels.txt")
    features = np.loadtxt(loadpath + "features_idx_list.txt", dtype=int)

    criterion = CRITERION

    if criterion == "Si":
        tag = "first-order"
    elif criterion == "STi":
        tag = "total"

    if features.shape == ():
        features_list = [features]
    else:
        features_list = list(features)

    msg = f"\nParameters ranking will be performed according to {tag} effects on selected features:\n"
    for idx in features_list:
        msg += f" {label[idx]}"
    print(msg)

    emul_type = EMUL_TYPE
    metric_name = WATCH_METRIC
    metric_type = HIGHEST_IS_BEST
    thre = THRE

    r_dct = {key: [] for key in index_i}

    for idx in features_list:
        path = loadpath_sobol + f"{idx}/"

        if not os.path.exists(path):
            raise Exception('Cannot find GPE folder '+loadpath_sobol+'/'+str(idx))

        if emul_type == "best":
            metric_score_list = np.loadtxt(
                path + metric_name + "_cv.txt", dtype=float
            )
            if metric_type:
                best_split = np.argmax(metric_score_list)
            else:
                best_split = np.argmin(metric_score_list)
            path += f"{best_split}/"

        S = np.loadtxt(path + criterion + ".txt", dtype=float)
        # S = correct_S(S, thre)

        mean = np.array([S[:, i].mean() for i in range(len(index_i))])
        ls = list(np.argsort(mean))

        for i, idx in enumerate(ls):
            if mean[idx] != 0:
                r_dct[index_i[idx]].append(i)
            else:
                r_dct[index_i[idx]].append(0)

    for key in r_dct.keys():
        r_dct[key] = -np.sum(r_dct[key])

    tuples = sorted(r_dct.items(), key=lambda tup: tup[1])

    R = {}
    c = 0
    for i, t in enumerate(tuples):
        if t[1] == 0:
            if c == 0:
                i0 = i + 1
                if i < len(index_i) - 1:
                    c += 1
            else:
                c += 1
            rank = f"#{i0}_{c}"
        else:
            rank = f"#{i+1}"

        R[rank] = t[0]

    print(
        f"\nParameters ranking from the most to the least important is:\n {R}"
    )


def gsa_parameters_ranking_S(loadpath,
                             loadpath_sobol,
                             gsa_mode="STi",
                             mode="max",
                             threshold_cutoff=0.9,
                             output_file=None,
                             features_file=None,
                             important_params_idx_file=None,
                             uncertainty=True):

    """
    Ranks parameters using sensitivity indies.

    Args:
        - loadpath: datafolder containing data used for GPE training
        - loadpath_sobol: folder containing GPEs and GSA results
        - gsa_mode: output GSA file to read in [STi, S1]
        - mode: how to rank the parameters [max,mean,sum]
        - threshold_cutoff: to decide which parameters are important
        - output_file: output file to save ranking
        - features_file: features file containing which features to consider
        - important_params_idx_file: where to save the indices of the important parameters
        - uncertainty: if True, the GSA had to be run accounting for GPE uncertainty,
                       e.g. by sampling the posterior distribution of the GPEs N=1000 times
    """

    print('Checking folder structure...')
    to_check = [loadpath + "xlabels.txt",
                loadpath + "ylabels.txt",
                loadpath_sobol]
    for f in to_check:
        if not os.path.exists(f):
            raise Exception('Cannot find file '+f)
        else:
            print(f+' found.')

    xlabels = read_labels(loadpath+"xlabels.txt")
    ylabels = read_labels(loadpath+"ylabels.txt")

    if features_file is None:
        features_file = loadpath + "features_idx_list.txt"

    if not os.path.exists(features_file):
        raise Exception('Cannot find '+features_file)

    features = np.loadtxt(features_file, dtype=int)
    if len(features.shape)==0:
        features = [features]
    else:
        features = list(features)

    print('Ranking parameters using features:')
    for idx in features:
        print(ylabels[idx])

    S = np.zeros((len(xlabels),len(features)),dtype=float)
    for i,idx in enumerate(features):
        if not os.path.exists(loadpath_sobol+str(idx)+"/"+gsa_mode+".txt"):
            raise Exception('Cannot find GSA file '+loadpath_sobol+str(idx)+"/"+gsa_mode+".txt")
        S_idx = np.loadtxt(loadpath_sobol+str(idx)+"/"+gsa_mode+".txt")
        if uncertainty:
            S[:,i] = np.mean(S_idx, axis=0)
        else:
            S[:,i] = S_idx

    S_total = np.zeros((len(xlabels),1),dtype=float)
    print('Ranking parameters according to their '+mode+' effect...')
    if mode=="mean":
        S_total = np.mean(S,axis=1)
    elif mode=="sum":
        S_total = np.sum(S,axis=1)
    elif mode=="max":
        S_total = np.max(S,axis=1)
    else:
        print("mode not recognised: please choose between mean, max and sum")

    ranked = np.argsort(S_total)
    ranked = ranked[::-1]
    ranked_S = S_total[ranked]

    if output_file is None:
        output_file = loadpath_sobol+"Rank_"+gsa_mode+"_"+mode+".txt"

    f = open(output_file, "w")
    for i in range(len(xlabels)):
        f.write(xlabels[ranked[i]]+"\t"+str(ranked_S[i])+"\n")
    f.close()

    print('Normalising ranked sensitivity to compute explained variance...')
    ranked_S_norm = list(np.array(ranked_S)/sum(ranked_S))

    ranked_S_norm_cumulative = []
    for i in range(len(xlabels)):
        ranked_S_norm_cumulative.append(sum(ranked_S_norm[0:i+1]))

    if output_file is None:
        output_file = loadpath_sobol+"Rank_"+gsa_mode+"_"+mode+"_ExpVariance.txt"
    else:
        output_file = output_file[:-4]+"_ExpVariance.txt"

    f = open(output_file, "w")
    for i in range(len(xlabels)):
        f.write(xlabels[ranked[i]]+"\t"+str(ranked_S_norm[i])+"\t"+str(ranked_S_norm_cumulative[i])+"\n")
    f.close()

    if important_params_idx_file is not None:
        idx_cutoff = np.where(np.array(ranked_S_norm_cumulative)>threshold_cutoff)[0][0]
        idx_param = ranked[range(idx_cutoff+1)]
        np.savetxt(important_params_idx_file,idx_param,fmt="%g") 

def gsa_parameters_ranking_S_union(loadpath,
                                   loadpath_sobol,
                                   gsa_mode="STi",
                                   mode="max",
                                   threshold_cutoff=0.9,
                                   output_file=None,
                                   features_file=None,
                                   important_params_idx_file=None):

    """
    Ranks parameters using sensitivity indies.

    Args:
        - loadpath: datafolder containing data used for GPE training
        - loadpath_sobol: folder containing GPEs and GSA results
        - gsa_mode: output GSA file to read in 
        - mode: how to rank the parameters [max,mean,sum]
        - output_file: output file to save ranking
        - features_file: features file containing which features to consider
    """

    print('Checking folder structure...')
    to_check = [loadpath + "xlabels.txt",
                loadpath + "ylabels.txt",
                loadpath_sobol]
    for f in to_check:
        if not os.path.exists(f):
            raise Exception('Cannot find file '+f)
        else:
            print(f+' found.')

    xlabels = read_labels(loadpath+"xlabels.txt")
    ylabels = read_labels(loadpath+"ylabels.txt")

    if features_file is None:
        features_file = loadpath + "features_idx_list.txt"

    if not os.path.exists(features_file):
        raise Exception('Cannot find '+features_file)

    features = np.loadtxt(features_file, dtype=int)
    if len(features.shape)==0:
        features = [features]
    else:
        features = list(features)

    print('Ranking parameters using features:')
    for idx in features:
        print(ylabels[idx])

    S = np.zeros((len(xlabels),len(features)),dtype=float)
    for i,idx in enumerate(features):
        if not os.path.exists(loadpath_sobol+str(idx)+"/"+gsa_mode+".txt"):
            raise Exception('Cannot find GSA file '+loadpath_sobol+str(idx)+"/"+gsa_mode+".txt")
        S_idx = np.loadtxt(loadpath_sobol+str(idx)+"/"+gsa_mode+".txt")
        S[:,i] = np.mean(S_idx, axis=0)

    print('Looking for 90% for each output...')

    dct = {}
    important = []
    for i, idx in enumerate(features):
        dct[ylabels[idx]] = {}
        dct[ylabels[idx]]["90%"] = {}
        dct[ylabels[idx]]["unimportant"] = {}

        S_tmp = S[:,i]

        ranked = np.argsort(S_tmp)
        ranked = ranked[::-1]

        ranked_S = S_tmp[ranked]
        ranked_S_norm = list(np.array(ranked_S)/sum(ranked_S))
        ranked_S_norm_cum = np.array([sum(ranked_S_norm[0:j+1]) for j in range(len(xlabels))],dtype=float)

        idx_last = np.where(np.array(ranked_S_norm_cum)>0.9)[0][0]
        important_tmp = ranked[:idx_last]

        xlabels_ranked = [xlabels[j] for j in ranked]
        for j,r in enumerate(ranked):
            if j<= idx_last:
                dct[ylabels[idx]]["90%"][xlabels_ranked[j]] = ranked_S_norm_cum[j]
            else:
                dct[ylabels[idx]]["unimportant"][xlabels_ranked[j]] = ranked_S_norm_cum[j]

        important += list(important_tmp)

    important = np.unique(np.array(important))

    dct["90%"] = {}
    for j in important:
        dct["90%"][xlabels[j]] = j

    write_json(dct,output_file)

    # S_total = np.zeros((len(xlabels),1),dtype=float)
    # print('Ranking parameters according to their '+mode+' effect...')
    # if mode=="mean":
    #     S_total = np.mean(S,axis=1)
    # elif mode=="sum":
    #     S_total = np.sum(S,axis=1)
    # elif mode=="max":
    #     S_total = np.max(S,axis=1)
    # else:
    #     print("mode not recognised: please choose between mean, max and sum")

    # ranked = np.argsort(S_total)
    # ranked = ranked[::-1]
    # ranked_S = S_total[ranked]

    # if output_file is None:
    #     output_file = loadpath_sobol+"Rank_"+gsa_mode+"_"+mode+".txt"

    # f = open(output_file+".txt", "w")
    # for i in range(len(xlabels)):
    #     f.write(xlabels[ranked[i]]+"\t"+str(ranked_S[i])+"\n")
    # f.close()

    # print('Normalising ranked sensitivity to compute explained variance...')
    # ranked_S_norm = list(np.array(ranked_S)/sum(ranked_S))

    # ranked_S_norm_cumulative = []
    # for i in range(len(xlabels)):
    #     ranked_S_norm_cumulative.append(sum(ranked_S_norm[0:i+1]))

    # if output_file is None:
    #     output_file = loadpath_sobol+"Rank_"+gsa_mode+"_"+mode+"_ExpVariance.txt"

    # f = open(output_file+"_ExpVariance.txt", "w")
    # for i in range(len(xlabels)):
    #     f.write(xlabels[ranked[i]]+"\t"+str(ranked_S_norm[i])+"\t"+str(ranked_S_norm_cumulative[i])+"\n")
    # f.close()

    # if important_params_idx_file is not None:
    #     idx_cutoff = np.where(np.array(ranked_S_norm_cumulative)>threshold_cutoff)[0][0]
    #     idx_param = ranked[range(idx_cutoff+1)]
    #     np.savetxt(important_params_idx_file,idx_param,fmt="%g") 

def match_lists(list1,list2):
    matched_idx = np.zeros((len(list1),),dtype=int)
    for i1,l1 in enumerate(list1):
        for i2,l2 in enumerate(list2):
            if l1==l2:
                matched_idx[i1] = i2

    return matched_idx

def gsa_select_important_param(loadpath,
                               loadpath_sobol,
                               gsa_mode="STi",
                               mode="max",
                               var_th=0.9):

    """
    Select important parameters according to explained variance.

    Args:
        - loadpath: datafolder containing data used for GPE training
        - loadpath_sobol: folder containing GPEs and GSA results
        - gsa_mode: output GSA file to read in 
        - mode: how to rank the parameters [max,mean,sum]
        - var_th: variance to explain to select important parameters
    """

    if var_th<0 or var_th>1:
        raise Exception('Provide a threshold for explained variance between 0 and 1.')

    print('Checking folder structure...')
    to_check = [loadpath + "/xlabels.txt",
                loadpath + "/default.txt",
                loadpath_sobol,
                loadpath_sobol+"/Rank_"+gsa_mode+"_"+mode+"_ExpVariance.txt"]
    for f in to_check:
        if not os.path.exists(f):
            raise Exception('Cannot find file '+f)
        else:
            print(f+' found.')

    xlabels = read_labels(loadpath+"/xlabels.txt")
    default = np.loadtxt(loadpath+"/default.txt")

    f = open(loadpath_sobol+"/Rank_"+gsa_mode+"_"+mode+"_ExpVariance.txt","r")
    lines = f.readlines()

    print('Selecting important parameters to explain '+str(var_th)+' of variance.')

    r_dct = {}
    count = 0
    for line in lines:
        line_split = re.split(r'\t+', line)
        r_dct[line_split[0]] = float(line_split[2])
        count += 1

    dct_idx = match_lists(r_dct.keys(),xlabels)

    cumulative_S = []
    for l in r_dct.keys():
        cumulative_S.append(r_dct[l])

    cutoff = np.where(np.array(cumulative_S)>var_th)[0][0]

    f = open(loadpath_sobol+"xlabels_Var"+str(var_th)+".txt", "w")
    f_d = open(loadpath_sobol+"default_Var"+str(var_th)+".txt", "w")
    for i in range(cutoff+1):
        f.write(xlabels[dct_idx[i]]+"\n")
        f_d.write(str(default[dct_idx[i]])+"\n")
    f.close()
    f_d.close()

