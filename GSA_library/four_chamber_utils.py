import os

import copy

import numpy as np

from SALib.sample import sobol_sequence
from SALib.util import scale_samples, nonuniform_scale_samples, read_param_file, compute_groups_matrix

from gpytGPE.gpe import GPEmul

from Historia.shared.design_utils import get_minmax, lhd, read_labels
from Historia.history import hm

from GSA_library.file_utils import read_json

def find_NIMP_samples(X,
					  settings,
					  memory_care=False):

	"""
	Given a matrix X of parameter combinations, return the index 
	of the samples that are acceptable according to the wave of 
	history matching ran with the settings given in the 
	settings dictionary

	Args:
		- X: set of parameters to analyse
		- settings: dictionary giving the path of the wave to use
					and which GPEs
		- memory_care: when we need to evaluate the GPEs on too many samples, the 
					   computer goes out of memory. If you set memory_care to True,
					   the vector of samples is split into parts to avoid memory issues

	Outputs: 
		- nimp_idx_final: indices of the acceptable samples

	"""

	W = hm.Wave()
	W.load(settings["wave_file"])   

	print('-----------------------------')
	print('Loading emulators for wave...')
	print('-----------------------------')  

	emulator_w = []
	for idx_w in settings["wave_features_idx"]:

		loadpath_wave = settings["wave_gpes_path"] + str(idx_w) + "/"

		X_train_w = np.loadtxt(loadpath_wave + "X_train.txt", dtype=np.float64)
		y_train_w = np.loadtxt(loadpath_wave + "y_train.txt", dtype=np.float64)

		emul_w = GPEmul.load(X_train_w, y_train_w, loadpath=loadpath_wave)
		emulator_w.append(emul_w)

	W.emulator = emulator_w 

	W.find_regions(X,memory_care=memory_care)
	nimp_idx = W.nimp_idx

	if "wave_gpes_path_final" in settings:

		print('Refining NIMP with better GPEs...')
		X_new = X[nimp_idx,:]
		
		emulator_w_refined = []
		for idx_w in settings["wave_features_idx"]:	

			loadpath_wave = settings["wave_gpes_path_final"] + str(idx_w) + "/"	

			X_train_w = np.loadtxt(loadpath_wave + "X_train.txt", dtype=np.float64)
			y_train_w = np.loadtxt(loadpath_wave + "y_train.txt", dtype=np.float64)	

			emul_w = GPEmul.load(X_train_w, y_train_w, loadpath=loadpath_wave)
			emulator_w_refined.append(emul_w)

		W.emulator = emulator_w_refined 

		W.find_regions(X_new,memory_care=memory_care)
		nimp_idx_final = nimp_idx[W.nimp_idx]

	else:
		nimp_idx_final = nimp_idx

	return nimp_idx_final

def lhd_gpes(X_init_file,
			 N,
			 waves_settings,
			 samples_file,
			 memory_care=False):

	"""
	Computes a latin hypercube sampling by restricting subsets of parameters
	with sub-model GPEs (ToR-ORd-Land, Courtemanche-Land, electrophysiology).
	This is designed in particular to start the history matching procedure
	for a set of electromechanics four-chamber simulations.

	Args:
		- X_init_file: points that were used to train the GPEs for the 
					   four-chamber outputs. This is needed to get the
					   parameter ranges
		- N: how many acceptable samples you want at the end
		- waves_settings: json file for the wave settings.
		- base_sequence_file: output file containing the samples or file 
							  containing the base sequence samples
		- samples_file: file where to save the samples
		- memory_care: when we need to evaluate the GPEs on too many samples, the 
					   computer goes out of memory. If you set memory_care to True,
					   the vector of samples is split into parts to avoid memory issues

	"""

	do = True
	if os.path.exists(samples_file):
		redo = input('The output file exists already. Do you want to overwrite? [y/n]')
		if redo!='y':
			do = False

	if do:
		settings = read_json(waves_settings)	

		X_init = np.loadtxt(X_init_file,dtype=float)
		I = get_minmax(X_init)	

		to_adapt = list(settings.keys())	

		X_test = lhd(I,N)	

		for f in to_adapt:
			print('Finding samples for '+f+'...')		

			N_final = 0
			factor = 1
			while N_final<N:	

				X_tmp = lhd(I[settings[f]["wave_idx_param"],:],N*factor)
			
				nimp_idx = find_NIMP_samples(X_tmp,
							  	             settings[f],
							  	             memory_care=memory_care)		

				N_final = nimp_idx.shape[0]	

				print('Found N='+str(N_final)+' acceptable samples.')		

				if N_final<N:
					factor *= 2	
					print('Increasing LHD N to '+str(N*factor)+'.')
				else:
					X_test[:,settings[f]["wave_idx_param"]] = X_tmp[nimp_idx[:N],]	

		np.savetxt(samples_file,X_test,fmt="%g")

def lhd_base_sequence(X_init_file,
			 		  N,
			 		  waves_settings,
			 		  base_sequence_file,
			 		  memory_care=False):

	"""
	Computes a base sequence for the Saltelli sampling
	using a Latin hypercube design. This assumes that a subset of parameters
	(given in waves_settings) were constrained before the simulations were run.
	This is the case for instance for the ToR-ORd-Land model, where we had to exclude
	samples where the tension was below a certain peak, and the rest tension was 
	above 1kPa/2kPa for stretch 1.0/1.1. In this function, we use the GPEs trained 
	on the cell model to get rid of the implausible samples at the cellular level.

	wave_settings provides: the field name (ToRORd_land, COURTEMANCHE_land, ...),
	the idx of the parameters that need to be constrained with each submodel
	and the wave and the GPEs we need to use to constrain the parameters

	Args:
		- X_init_file: points that were used to train the GPEs for the 
					   four-chamber outputs. This is needed to get the
					   parameter ranges
		- N: how many acceptable samples you want at the end
		- waves_settings: json file for the wave settings.
		- base_sequence_file: output file containing the samples or existing file 
							  containing the samples
		- memory_care: when we need to evaluate the GPEs on too many samples, the 
					   computer goes out of memory. If you set memory_care to True,
					   the vector of samples is split into parts to avoid memory issues

	"""

	do = True
	if os.path.exists(base_sequence_file):
		redo = input('The output file exists already. Do you want to overwrite? [y/n]')
		if redo!='y':
			do = False

	if do:
		settings = read_json(waves_settings)	

		X_init = np.loadtxt(X_init_file,dtype=float)
		I = get_minmax(X_init)	

		D = X_init.shape[1]

		to_adapt = list(settings.keys())	

		I_sequence = np.concatenate((I,I),axis=0)
		X_test = lhd(I_sequence,N)	

		base_sequence = lhd(I_sequence,N)

		for f in to_adapt:
			print('Finding samples for '+f+'...')		

			N_final = 0
			factor = 1
			while N_final<N:	
				D_tmp = len(settings[f]["wave_idx_param"])
				param_idx = settings[f]["wave_idx_param"]
				I_tmp = np.concatenate((I[param_idx,:],I[param_idx,:]),axis=0)

				base_sequence_initial = lhd(I_tmp,N*factor)
				A = base_sequence_initial[:,:D_tmp]
				B = base_sequence_initial[:,D_tmp:]
				A_rescaled = copy.deepcopy(A)
				B_rescaled = copy.deepcopy(B)   

				A_nimp_idx = find_NIMP_samples(A_rescaled,
							  	             settings[f],
							  	             memory_care=memory_care)		

				B_nimp_idx = find_NIMP_samples(B_rescaled,
							  	             settings[f],
							  	             memory_care=memory_care)		

				nimp_idx = np.intersect1d(A_nimp_idx,B_nimp_idx)

				N_final = nimp_idx.shape[0]	

				print('Found N='+str(N_final)+' acceptable samples.')		

				if N_final<N:
					factor *= 2	
					print('Increasing Sobol N to '+str(N*factor)+'.')
				else:
					base_sequence[:,settings[f]["wave_idx_param"]] = A[nimp_idx[:N],:]
					base_sequence[:,np.array(settings[f]["wave_idx_param"])+D] = B[nimp_idx[:N],:]

		base_sequence_normalised = copy.deepcopy(base_sequence)
		for i in range(I_sequence.shape[0]):
			base_sequence_normalised[:,i] = (base_sequence[:,i]-I_sequence[i,0])/(I_sequence[i,1]-I_sequence[i,0])

		np.savetxt(base_sequence_file,base_sequence_normalised,fmt="%g")

def Sobol_gpes(X_init_file,
   			   N,
   			   waves_settings,
   			   base_sequence_file,
   			   saltelli_sequence_file,
   			   saltelli_weigths_file,
   			   memory_care=False,
   			   calc_second_order=False):

	"""
	Computes a base sequence for the Saltelli sampling
	using a Latin hypercube design. This assumes that a subset of parameters
	(given in waves_settings) were constrained before the simulations were run.
	This is the case for instance for the ToR-ORd-Land model, where we had to exclude
	samples where the tension was below a certain peak, and the rest tension was 
	above 1kPa/2kPa for stretch 1.0/1.1. In this function, we use the GPEs trained 
	on the cell model to get rid of the implausible samples at the cellular level.

	wave_settings provides: the field name (ToRORd_land, COURTEMANCHE_land, ...),
	the idx of the parameters that need to be constrained with each submodel
	and the wave and the GPEs we need to use to constrain the parameters

	Args:
		- X_init_file: points that were used to train the GPEs for the 
					   four-chamber outputs. This is needed to get the
					   parameter ranges
		- N: how many acceptable samples you want at the end
		- waves_settings: json file for the wave settings.
		- base_sequence_file: output file containing the samples or file 
							  containing the base sequence samples
		- saltelli_sequence_file: where to save the saltelli sequence created from 
								  the starting base sequence
		- saltelli_weigths_file: file containing a 1 where the Saltelli sample is 
								 acceptable, 0 otherwise. The Saltelli sequence
								 is created by 'mixing and matching' the columns of the
								 base sequence. So the fact that all samples in the base
								 sequence are acceptable, does not imply the same for the 
								 Saltelli samples. These are therefore checked again with the
								 GPEs of each submodel and the binary weight saved in this file.
		- memory_care: when we need to evaluate the GPEs on too many samples, the 
					   computer goes out of memory. If you set memory_care to True,
					   the vector of samples is split into parts to avoid memory issues

	"""

	do = True
	if os.path.exists(base_sequence_file):
		redo = input('The Sobol sequence file exists already. Do you want to overwrite? [y/n]')
		if redo!='y':
			do = False

	print('!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!')
	print('                   Setting skip_values to 0                ')
	print('!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!')

	skip_values = 0

	X_init = np.loadtxt(X_init_file,dtype=float)
	D = X_init.shape[1]
	I = get_minmax(X_init)	

	settings = read_json(waves_settings)	
	to_adapt = list(settings.keys())	

	if do:

		base_sequence = sobol_sequence.sample(N + skip_values, 2 * D)

		for f in to_adapt:
			print('Finding samples for '+f+'...')		

			N_final = 0
			factor = 1
			while N_final<N:	
				D_tmp = len(settings[f]["wave_idx_param"])

				base_sequence_initial = sobol_sequence.sample(N*factor + skip_values, 2 * D_tmp)
				A = base_sequence_initial[:,:D_tmp]
				B = base_sequence_initial[:,D_tmp:]
				A_rescaled = copy.deepcopy(A)
				B_rescaled = copy.deepcopy(B)   

				for i in range(D_tmp):
					A_rescaled[:,i] = A_rescaled[:,i]*(I[settings[f]["wave_idx_param"][i],1]-I[settings[f]["wave_idx_param"][i],0])+I[settings[f]["wave_idx_param"][i],0]
					B_rescaled[:,i] = B_rescaled[:,i]*(I[settings[f]["wave_idx_param"][i],1]-I[settings[f]["wave_idx_param"][i],0])+I[settings[f]["wave_idx_param"][i],0]   
			
				A_nimp_idx = find_NIMP_samples(A_rescaled,
							  	             settings[f],
							  	             memory_care=memory_care)		

				B_nimp_idx = find_NIMP_samples(B_rescaled,
							  	             settings[f],
							  	             memory_care=memory_care)		

				nimp_idx = np.intersect1d(A_nimp_idx,B_nimp_idx)

				N_final = nimp_idx.shape[0]	

				print('Found N='+str(N_final)+' acceptable samples.')		

				if N_final<N:
					factor *= 2	
					print('Increasing Sobol N to '+str(N*factor)+'.')
				else:
					base_sequence[:,settings[f]["wave_idx_param"]] = A[nimp_idx[:N],:]
					base_sequence[:,np.array(settings[f]["wave_idx_param"])+D] = B[nimp_idx[:N],:]

		np.savetxt(base_sequence_file,base_sequence,fmt="%g")

	else:

		base_sequence = np.loadtxt(base_sequence_file,dtype=float)

	Dg = D
	groups = False

	if calc_second_order:
		saltelli_sequence = np.zeros([(2 * Dg + 2) * N, D])
	else:
		saltelli_sequence = np.zeros([(Dg + 2) * N, D])
	index = 0

	for i in range(skip_values, N + skip_values):

		# Copy matrix "A"
		for j in range(D):
			saltelli_sequence[index, j] = base_sequence[i, j]

		index += 1

		# Cross-sample elements of "B" into "A"
		for k in range(Dg):
			for j in range(D):
				if (not groups and j == k) or (groups and group_names[k] == groups[j]):
					saltelli_sequence[index, j] = base_sequence[i, j + D]
				else:
					saltelli_sequence[index, j] = base_sequence[i, j]

			index += 1

		# Cross-sample elements of "A" into "B"
		# Only needed if you're doing second-order indices (true by default)
		if calc_second_order:
			for k in range(Dg):
				for j in range(D):
					if (not groups and j == k) or (groups and group_names[k] == groups[j]):
						saltelli_sequence[index, j] = base_sequence[i, j]
					else:
						saltelli_sequence[index, j] = base_sequence[i, j + D]

				index += 1

		# Copy matrix "B"
		for j in range(D):
			saltelli_sequence[index, j] = base_sequence[i, j + D]

		index += 1

	scale_samples(saltelli_sequence, I)
	weights = np.zeros((saltelli_sequence.shape[0],),dtype=int)    

	nimp_idx = []
	for f in to_adapt:

		print('--------------------------------------------------------------------------------')
		print('Scanning A & B crossed samples to make sure they are in the NIMP for '+f+'...')
		print('--------------------------------------------------------------------------------')
		print('\n')

		nimp_idx_tmp = find_NIMP_samples(saltelli_sequence[:,settings[f]["wave_idx_param"]],
					  	             settings[f],
					  	             memory_care=memory_care)		
		if len(nimp_idx)==0:
			nimp_idx = nimp_idx_tmp
		else:
			nimp_idx = np.intersect1d(np.array(nimp_idx),nimp_idx_tmp)

	N_final = nimp_idx.shape[0]	

	print('Found '+str(N_final)+' samples.')
	print('Saving samples and weigths...')

	weights[nimp_idx] = 1

	np.savetxt(saltelli_sequence_file,saltelli_sequence,fmt="%g")
	np.savetxt(saltelli_weigths_file,weights,fmt="%d")