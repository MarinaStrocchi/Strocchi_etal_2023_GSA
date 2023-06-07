import sys

import json

import numpy as np
import pandas as pd

from itertools import combinations
from scipy.special import binom
import random

import matplotlib.gridspec as grsp
import matplotlib.pyplot as plt
import seaborn as sns

from SALib.sample import morris as ms
from SALib.analyze import morris as ma
import timeit
import torch

from gpytGPE.utils.design import lhd
from gpytGPE.utils.design import read_labels

from GSA_library.file_utils import read_json, write_json

def OAT_sample_circadapt(range_int=0.2,
						 basefolder='./test'):

	"""
	Generates samples for a one-at-a-time sensitivity analysis on the 
	CircAdapt ODEs model. The code writes in output one json file
	per parameter combination to be given in input to the CircAdapt ODEs solver
	in CARP, and the matrix X.txt.

	Args:
		range_int: percentage change of each parameter
		basefolder: basefolder containing data/, json/default.json, json/params_name.txt
	"""

	files_to_check = [basefolder+"/data/",
					  basefolder+'/json/default.json',
					  basefolder+'/json/params_name.txt']

	for f in files_to_check:
		if not os.path.exists(f):
			raise Exception("Cannot find file "+f+".")

	filename=basefolder+'/json/default.json'
	p = read_json(filename)

	params_name = read_labels(basefolder+'/json/params_name.txt')

	default_ = [p[key] for key in params_name]
	
	H = np.zeros((len(default_)+1,len(default_)))
	H[0,:] = default_
	for i in range(len(default_)):
		H[i+1,:] = default_
		p_temp = p
		if params_name[i] == 'VV delay [s]':
			p_temp[params_name[i]] = 0.02
			H[i+1,i] = 0.02
		else:
			p_temp[params_name[i]] = default_[i]*(1.+range_int)
			H[i+1,i] = H[i+1,i]*(1.+range_int)
		p_temp['RV max isometric stress [Pa]'] = p_temp['LV max isometric stress [Pa]']
		p_temp['RA max isometric stress [Pa]'] = p_temp['LA max isometric stress [Pa]']
		p_temp['RV passive stress [Pa]'] = p_temp['LV passive stress [Pa]']
		p_temp['RA passive stress [Pa]'] = p_temp['LA passive stress [Pa]']
		write_json(p_temp,basefolder+'/json/'+str(i+1)+'.json')

	np.savetxt(basefolder+'/data/X.txt', H, fmt='%.4f')	