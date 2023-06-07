import os
import sys

import json

import numpy as np
import pandas as pd

from itertools import combinations
import random

import matplotlib.gridspec as grsp
import matplotlib.pyplot as plt
import seaborn as sns

from SALib.sample import morris as ms
from SALib.analyze import morris as ma
from scipy.special import binom
import timeit
import torch

from gpytGPE.utils.design import lhd
from gpytGPE.utils.design import read_labels

from GSA_library.file_utils import read_json, write_json

SEED=8

def morris_sample_circadapt(num_levels=4,
							num_trajectories=4,
							range_int=0.2,
							basefolder='./test'):

	"""
	Generate a Morris sample for the CircAdapt model. This generates 
	one json file per parameter combination to input to the CircAdapt
	ODEs solver in CARP.

	Args:
		- num_levels: number of levels in the Morris sampling
		- num_trajectories: how many trajectories to generate
		- range_int: percentage range to change the parameters (default+/-range*default)
		- basefolder: folder containing the json/default.json and json/params_name.txt,
					  data/xlabels.txt

	"""

	if not os.path.exists(basefolder+'/json/default.json'):
		raise Exception("You need to have a json folder in basefolder, containing a default.json file.")

	filename=basefolder+'/json/default.json'
	p = read_json(filename)

	if not os.path.exists(basefolder+'/json/params_name.txt'):
		raise Exception("You need to have a json folder in basefolder, containing a params_name.txt to specify which parameters you want to change.")

	params_name = read_labels(basefolder+'/json/params_name.txt')

	default_ = [p[key] for key in params_name]
	R = np.zeros((len(default_),2))
	for i,pp in enumerate(default_):
		if params_name[i] == 'VV delay [s]':
			R[i,:] = np.array([-0.02,0.02])
		else:
			R[i,:] = np.array([pp*(1.-range_int),pp*(1.+range_int)])

	N = 1000 # number of trajectories to generate and then take the optimal ones
	local_optimization = True

	if not os.path.exists(basefolder+'/data/xlabels.txt'):
		raise Exception("You need to have a data folder in basefolder, containing an xlabels.txt file.")

	index_i = read_labels(basefolder + '/data/xlabels.txt')
	index_ij = ['({}, {})'.format(c[0], c[1]) for c in combinations(index_i, 2)]

	D = len(params_name)
	problem = {
		'num_vars': D,
		'names': index_i,
		'bounds': R.tolist()
	}

	H = ms.sample(problem, N, num_levels=num_levels, 
							  optimal_trajectories=num_trajectories, 
							  local_optimization=True, 
							  seed=SEED) 
	n_samples = np.size(H[:,0])
	for i in range(n_samples):
		p_temp = p
		for j in range(len(default_)):
			if params_name[j] == 'Mean systemic flow at rest [m3/s]':
				p_temp['Mean systemic flow at rest [m3/s]'] = int(H[i,j]*1e07)*1e-07
			else:
				p_temp[params_name[j]] = round(H[i,j],3)
		p_temp['RV max isometric stress [Pa]'] = p_temp['LV max isometric stress [Pa]']
		p_temp['RA max isometric stress [Pa]'] = p_temp['LA max isometric stress [Pa]']
		p_temp['RV passive stress [Pa]'] = p_temp['LV passive stress [Pa]']
		p_temp['RA passive stress [Pa]'] = p_temp['LA passive stress [Pa]']
		write_json(p_temp,basefolder+'/json/'+str(i+1)+'.json')

	problem['num_levels'] = num_levels
	problem['optimal_trajectories'] = num_trajectories
	problem_j = json.dumps(problem)
	f = open(basefolder+'/data/Morris_problem.json','w')
	f.write(problem_j)
	f.close()

	np.savetxt(basefolder+'/data/X.txt', H, fmt='%.4f')	

def morris_gsa_circadapt(idx_feature=0,
					     chamber='LV',
					     basefolder='./examples/circadapt'):

	"""
	To run a Morris sensitivity analysis given a specific feature.

	Args:
		- idx_feature: which feature to run the analysis for
		- chamber: which chamber (LV,RV,LA,RA)
		- basefolder: folder where to save and find data/X.txt,
					  chamber_output.txt

	"""

	path = basefolder+'/morris_gsa/' + chamber + '/' 
	#========================
	# define sample
	#========================
	start_time = timeit.default_timer()
	if (chamber == 'LV') or (chamber == 'RV'):
		labels_name = 'ylabels_v.txt'
	elif (chamber == 'LA') or (chamber == 'RA'):
		labels_name = 'ylabels_a.txt'

	files_to_check = [basefolder+"/data/"+labels_name,
					  basefolder+"/data/X.txt",
					  basefolder+'/data/'+chamber+'_output.txt',
					  basefolder+'/data/Morris_problem.json']

	for f in files_to_check:
		if not os.path.exists(f):
			raise Exception("Cannot find file "+f+".")

	X = np.loadtxt(basefolder+'/data/X.txt', dtype='f', delimiter=' ')

	Y = np.loadtxt(basefolder+'/data/'+chamber+'_output.txt', dtype='f', delimiter=' ')

	f = open(basefolder+'/data/Morris_problem.json')
	problem_j = json.load(f)

	problem = {
		'num_vars': problem_j['num_vars'],
		'names': problem_j['names'],
		'bounds': np.array(problem_j['bounds'])
	}
	num_params = problem_j['num_vars']
	num_levels = problem_j['num_levels']
	num_trajectories = problem_j['optimal_trajectories']
	n_samples = (num_params+1)*num_trajectories

	print(num_params)
	print(num_trajectories)
	print(Y.shape)
	print(n_samples)

	# remove outputs and inputs with crashed simulations
	sim_crash = []
	for i in range(n_samples):
		if np.sum(Y[i,:]) == 0:
			sim_crash.append(i)
	n_samples_filt = n_samples-len(sim_crash)
	print(str(n_samples-n_samples_filt)+'/'+str(n_samples)+' crashed...')
	
	if len(sim_crash)>0:
		ind_remove = []
		for i in range(len(sim_crash)):
			for d in range(num_trajectories):
				interval = [d*num_params+d,(d+1)*num_params+d]
				if (sim_crash[i]>=interval[0]) and (sim_crash[i]<=interval[1]):
					ind_remove.append(list(range(interval[0],interval[1]+1)))
		ind_remove_ = []
		for i in range(len(ind_remove)):
			for j in range(len(ind_remove[i])):
				if ind_remove[i][j] not in ind_remove_:
					ind_remove_.append(ind_remove[i][j])
		ind_ok = []	
		for i in range(n_samples):
			if i not in ind_remove:
				ind_ok.append(i)
		N = n_samples-len(ind_remove_)
		X_filt = X[ind_ok,:]
		Y_filt = Y[ind_ok,int(idx_feature)]
	else:
		N = n_samples
		X_filt = X
		Y_filt = Y[:,int(idx_feature)]
	print('Using '+str(N)+' simulations /'+str(n_samples_filt)+'...')

	cmd = 'mkdir -p '+path
	os.system(cmd)
	
    #=======================================
	mu =  np.zeros((1, N), dtype=float)
	mu_star =  np.zeros((1, N), dtype=float)
	sigma =  np.zeros((1, N), dtype=float)
	mu_star_conf =  np.zeros((1, N), dtype=float)

	S = ma.analyze(problem, X_filt, Y_filt, num_resamples=1000, conf_level=0.95, num_levels=num_levels, seed=SEED)
	mu = np.array(S['mu'])
	mu_star = np.array(S['mu_star'])
	sigma = np.array(S['sigma'])
	mu_star_conf = np.array(S['mu_star_conf'])

	print('GSA - Elapsed time: {:.4f} sec'.format(timeit.default_timer() - start_time))

	np.savetxt(path + 'feature_'+ str(idx_feature) + '_mu.txt', mu.reshape(1,problem['num_vars']), fmt='%.6f')
	np.savetxt(path + 'feature_'+ str(idx_feature) + '_mu_star.txt', mu_star.reshape(1,problem['num_vars']), fmt='%.6f')
	np.savetxt(path + 'feature_'+ str(idx_feature) + '_sigma.txt', sigma.reshape(1,problem['num_vars']), fmt='%.6f')
	np.savetxt(path + 'feature_'+ str(idx_feature) + '_mu_star_conf.txt', mu_star_conf.reshape(1,problem['num_vars']), fmt='%.6f')