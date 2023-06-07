import os

import numpy as np

from gpytGPE.utils.design import read_labels

from GSA_library import kfold_cross_validation_training
from GSA_library import global_sobol_sensitivity_analysis
from GSA_library import gsa_parameters_ranking

from GSA_library.plotting import *
from GSA_library.gsa_plotting import * 

PARALLEL=False
UNCERTAINTY=False

def main():

	basefolder = './example/'

	idx_feature = list(np.loadtxt(basefolder+"/data/features_idx_list.txt",dtype=int))

	# ================================================================
    # GPE TRAINING
    # ================================================================
	for idx in idx_feature:

		loadpath = basefolder + 'data/'
		savepath = basefolder + 'output/' + str(idx) + '/'

		if not os.path.exists(savepath+"gpe.pth"):
			cmd = 'mkdir -p ' + savepath
			os.system(cmd)	

			kfold_cross_validation_training.kfold_cross_validation_training(loadpath,
																			idx,
																			savepath,
																			parallel=PARALLEL)
		else:
			print("GPE "+savepath+"gpe.pth already found. Skipping training.")

    # ================================================================
    # GSA
    # ================================================================
	loadpath = basefolder + 'data/'
	for idx in idx_feature:

		savepath = basefolder + 'output/' + str(idx) + '/'
		if not os.path.exists(savepath+'/Si.txt'):

			global_sobol_sensitivity_analysis.global_sobol_sensitivity_analysis(loadpath,
																				idx,
																				savepath,
																				uncertainty=UNCERTAINTY)
		else:
			print("GSA for feature "+str(idx)+" already run. Skipping GSA.")

	# ================================================================
    # Param ranking
    # ================================================================
	loadpath = basefolder + 'data/'
	loadpath_sobol = basefolder + 'output/'
	gsa_parameters_ranking.gsa_parameters_ranking_S(loadpath,
												    loadpath_sobol,
												    gsa_mode="STi",
												    mode="max",
												    uncertainty=UNCERTAINTY)

	# ================================================================
    # PLOT
    # ================================================================

	loadpath = basefolder + 'data/'

	X = np.loadtxt(loadpath+'X.txt')
	Y = np.loadtxt(loadpath+'Y.txt')

	xlabels = read_labels(loadpath+'xlabels.txt')
	ylabels = read_labels(loadpath+'ylabels.txt')

	plotpath = basefolder + 'figures/'

	os.system("mkdir "+plotpath)

	gsapath = basefolder + 'output/'
	ST_all = np.zeros((len(xlabels),len(ylabels)),dtype=float)
	S1_all = np.zeros((len(xlabels),len(ylabels)),dtype=float)
	for i,idx in enumerate(idx_feature):
		ST = np.loadtxt(gsapath+str(idx)+'/STi.txt')
		S1 = np.loadtxt(gsapath+str(idx)+'/Si.txt')

		if UNCERTAINTY:
			ST_all[:,i] = np.mean(ST, axis=0)
			S1_all[:,i] = np.mean(S1, axis=0)
		else:
			ST_all[:,i] = ST
			S1_all[:,i] = S1

	gsa_heat(ST_all, S1_all, xlabels, ylabels, plotpath, correction=False,horizontal=True)

	plot_rank_GSA(loadpath,
				  loadpath_sobol,
				  criterion="STi",
				  mode="max",
				  figname=basefolder+"/figures/Rank_max.png",
				  th=0.0)

if __name__ == '__main__':
	main()