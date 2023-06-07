import os
import sys

import json

import numpy as np
import pandas as pd

import random
from itertools import combinations
from scipy.special import binom

import matplotlib.gridspec as grsp
import matplotlib.pyplot as plt
import seaborn as sns

from SALib.sample import morris as ms
from SALib.analyze import morris as ma
import torch
import timeit

from gpytGPE.utils.design import lhd
from gpytGPE.utils.design import read_labels

from GSA_library.file_utils import read_json, write_json

def X_to_eikonalInit(X,
					 json_tags,
					 json_param,
					 json_default,
					 output_folder,
					 electro_vtx_list,
					 stim_time_list):

	"""
	Converts an X matrix to init files for ekbatch simulation

	Args:
		- X: matrix of parameters to convert to init files
		- json_tags: json file containing the tags for eikonal simulation
		- json_param: json file containing the map between the tags and the columns of X.
					  Each columns of X corresponds to a conduction velocity assigned
					  to a set of tags.
		- output_folder: where to save the init files
		- electro_vtx_list: list of vtx files containing the stimuli
		- stim_time_list: time of stimuli for EVERY vtx file 

	"""

	if len(electro_vtx_list)!=len(stim_time_list):
		 raise Exception("Electrodes list and stimulus times must have the same length\n") 

	tags = read_json(json_tags)
	param_idx = read_json(json_param)
	default = read_json(json_default)

	ntags = 0
	tags_list = ["ventricles","fast_endo","atria","bachmann_bundle"]
	for t in tags_list:
		ntags += len(tags[t])

	for k in range(X.shape[0]):
		# ----------------------------------------------------------------------------------- #
		# write .init file
		f = open(output_folder+'/'+str(k)+'.init','w')	

		# header
		f.write('vf:1.000000 vs:1.000000 vn:1.000000 vPS:3.0\n')
		f.write('retro_delay:3.0 antero_delay:10.0\n')
		# number of stimuli and regions
		f.write('%d %d\n' % (len(electro_vtx_list),ntags))
		# stimulus
		for j in range(len(electro_vtx_list)):
			vtx = np.loadtxt(electro_vtx_list[j],skiprows=2,dtype=int)
			if not bool(vtx.shape):
				f.write('%d %f\n' % (vtx,stim_time_list[j]))
			else:
				for n in vtx:
					f.write('%d %f\n' % (int(n),stim_time_list[j]))

		# ek regions
		for j,t in enumerate(tags["ventricles"]):
			CV_f_v = X[k,param_idx["CV_f_v"]] if "CV_f_v" in param_idx else default["CV_f_v"]
			CV_s_v = CV_f_v*X[k,param_idx["ani_ratio_v"]] if "ani_ratio_v" in param_idx else CV_f_v*default["ani_ratio_v"]
			f.write('%d %f %f %f\n' % (int(t),CV_f_v,CV_s_v,CV_s_v))	
		for j,t in enumerate(tags["fast_endo"]):
			k_FEC = X[k,param_idx["k_FEC"]] if "k_FEC" in param_idx else default["k_FEC"]
			f.write('%d %f %f %f\n' % (int(t),CV_f_v*k_FEC,CV_s_v*k_FEC,CV_s_v*k_FEC))	
		for j,t in enumerate(tags["atria"]):
			CV_f_a = X[k,param_idx["CV_f_a"]] if "CV_f_a" in param_idx else default["CV_f_a"]
			CV_s_a = CV_f_a*X[k,param_idx["ani_ratio_a"]] if "ani_ratio_a" in param_idx else CV_f_a*default["ani_ratio_a"]
			f.write('%d %f %f %f\n' % (int(t),CV_f_a,CV_s_a,CV_s_a))	
		for j,t in enumerate(tags["bachmann_bundle"]):
			k_BB = X[k,param_idx["k_BB"]] if "k_BB" in param_idx else default["k_BB"]
			f.write('%d %f %f %f\n' % (int(t),CV_f_a*k_BB,CV_s_a*k_BB,CV_s_a*k_BB))	

		f.close()

def json_to_txt(json_file,
				params_name,
				out_txt):

	"""
	Converts json to file to txt file containing the parameter values

	Args:
		- json_file: json file containing parameter values
		- params_name: labels of the parameters
		- out_txt: output text file to write the parameter values

	"""

	if not os.path.exists(json_file):
		raise Exception('Cannot find '+json_file)

	p = read_json(json_file)
	params_name = read_labels(params_name)

	f = open(out_txt, "w")
	for i in range(len(params_name)):
		f.write(str(p[params_name[i]])+"\n")
	f.close()

def gpes_circadapt(N_samples=100,
				   range_int=0.2,
				   basefolder='./'):

	"""
	Samples the parameter space for the circadapt model
	given a default json file and a percentage range
	applied to all parameters.

	Args:
		- N_samples: number of samples
		- range_int: percentage interval to sample from default value 
					 (default value +/- range_int*default value)
		- basefolder: folder containinf default.json and the parameters names

	"""

	print('Checking folder structure...')
	to_check = [basefolder+'/json/',
				basefolder+'/json/default.json',
				basefolder+'/data/params_name.txt']
	for f in to_check:
		if not os.path.exists(f):
			raise Exception('Cannot find '+f)
		else:
			print(f+' found.')

	filename=basefolder+'/json/default.json'
	p = read_json(filename)

	params_name = read_labels(basefolder+'/data/params_name.txt')

	default_ = [p[key] for key in params_name]
	R = np.zeros((len(default_),2))
	for i,pp in enumerate(default_):
		if params_name[i] == 'VV delay [s]':
			R[i,:] = np.array([-0.02,0.02])
		else:
			R[i,:] = np.array([pp*(1.-range_int),pp*(1.+range_int)])

	H = lhd(R, N_samples)
	
	for i in range(N_samples):
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
		
		write_json(p_temp,basefolder+'/json/'+str(i)+'.json')

	np.savetxt(basefolder+'/data/X.txt', H, fmt='%g')	

def gpes_ionic(N_samples=100,
			   range_int=0.25,
			   basefolder='./',
			   param_range = [],
			   string_header='flags=ENDO'):

	"""
	Samples the parameter space for an ionic model
	given the basefolder, the number of samples and
	the percentage range to vary the parameters within.

	Args:
		- N_samples: number of samples
		- range_int: percentage interval to sample from default value 
						(default value +/- range_int*default value)
		- basefolder: folder containing the data folder and
						and the output folder for the param.txt
						files
		- param_range: optional additional ranges to use for some
						parameters instead of the specified range_int
						structured as [[name_1,name_2],[range_1,range_2]]
		- string_header: string to start the parameter string for

	"""

	if len(param_range)==0:
		print('Using the range '+str(range_int)+' for all parameters.')

	if range_int==0 or range_int>=1:
		raise Exception('Please specify a range between ]0,1]')

	print('Checking folder structure...')
	to_check = [basefolder+'/data/default.txt',
				basefolder+'/data/xlabels.txt']
	for f in to_check:
		if not os.path.exists(f):
			raise Exception('Cannot find '+f)
		else:
			print(f+' found.')

	if not os.path.exists(basefolder+'/param/'):
		cmd='mkdir '+basefolder+'/param/'
		os.system(cmd)

	p = np.loadtxt(basefolder+'/data/default.txt', dtype=float)
	params_name = read_labels(basefolder+'/data/xlabels.txt')

	R = np.zeros((len(params_name),2))
	for i,pp in enumerate(p):
		if len(param_range)>0:
			if params_name[i] in param_range[0]:
				for j,pr in enumerate(param_range[0]):
					if pr==params_name[i]:
						R[i,:] = np.array([pp*(1.-param_range[1][j]),pp*(1.+param_range[1][j])])
			else:
				R[i,:] = np.array([pp*(1.-range_int),pp*(1.+range_int)])
		else:
			R[i,:] = np.array([pp*(1.-range_int),pp*(1.+range_int)])
		if params_name[i] in ['ICaL_fractionSS','INaCa_fractionSS']:
			print(params_name[i]+' limited to a maximum of 1.0')
			R[i,1] = min(R[i,1],1.0)

	print('Generating '+str(N_samples)+' samples...');
	H = lhd(R, N_samples)
	
	print('Writing param.txt files for bench.pt simulations.')
	for i in range(N_samples):
		if string_header != '':
			string = string_header+','
		else:
			string = ''
		for j in range(len(params_name)-1):
			string += params_name[j]+'='+str(H[i,j])+','
		string += params_name[-1]+'='+str(H[i,-1])
		
		text_file = open(basefolder+'/param/'+str(i)+'_param.txt', 'w')
		n = text_file.write(string)
		text_file.close()
		
	np.savetxt(basefolder+'/data/X.txt', H, fmt='%.10f')

def gpes_EM(N_samples=100,
			range_int=0.25,
			datafolder='./',
			waveload = None,
			ionic='ToRORd_dynCl',
			string_header_ionic='flags=ENDO',
			string_header_land='',
			output_folder='./',
			param_range_ionic=[],
			param_range_land=[],
			ionic_sample='lhd'):

	"""
	Samples the parameter space for an ionic+land model
	given the basefolder, the number of samples and
	the percentage range to vary the parameters within.

	Args:
		- N_samples: number of samples
		- range_int: percentage interval to sample from default value 
						(default value +/- range_int*default value)
		- datafolder: folder containing the data folder and
						and the output folder for the param.txt
						files
		- param_range: optional additional ranges to use for some
						parameters instead of the specified range_int
						structured as [[name_1,name_2],[range_1,range_2]]
		- waveload: if the ionic model is sampled from a HM wave, 
					you need to provide the wavefolder
		- ionic: ionic model to use. Either ToRORd_dynCl or COURTEMANCHE
		- string_header_ionic: string to start the parameter string for
								ionic bench.pt simulation
		- string_header_land: string to start the parameter string for
								land bench.pt simulation
		- output_folder: where to save the samples
		- param_range_land: parameter-specific parameter range for the
			land model. [['name_1','name_2'],[range_1,range_2]]
		- ionic_sample: either 'HM' or 'lhd'

	"""

	print('Checking folder structure...')
	to_check = [datafolder+'/default_land.txt',
				datafolder+'/default_'+ionic+'.txt',
				datafolder+'/xlabels_'+ionic+'.txt']
	for f in to_check:
		if not os.path.exists(f):
			raise Exception('Cannot find '+f)
		else:
			print(f+' found.')

	p_land = np.loadtxt(datafolder+'/default_land.txt', dtype=float)
	params_name_land = read_labels(datafolder+'/xlabels_land.txt')
	N_land = len(params_name_land)

	p_ionic = np.loadtxt(datafolder+'/default_'+ionic+'.txt', dtype=float)
	params_name_ionic = read_labels(datafolder+'/xlabels_'+ionic+'.txt')
	N_ionic = len(params_name_ionic)


	if ionic_sample == 'lhd':
		print('Sampling with latin hypercube design both ep and mechanic models.')
		R_ionic = np.zeros((N_ionic,2))
		if N_ionic > 1:
			for i,pp in enumerate(p_ionic):
				if len(param_range_ionic)>0:
					if params_name_ionic[i] in param_range_ionic[0]:
						for j,pr in enumerate(param_range_ionic[0]):
							if pr==params_name_ionic[i]:
								R_ionic[i,:] = np.array([pp*(1.-param_range_ionic[1][j]),pp*(1.+param_range_ionic[1][j])])
					else:
						R_ionic[i,:] = np.array([pp*(1.-range_int),pp*(1.+range_int)])	
				else:
					R_ionic[i,:] = np.array([pp*(1.-range_int),pp*(1.+range_int)])	

				if params_name_ionic[i] in ['ICaL_fractionSS','INaCa_fractionSS']:
					print(params_name[i]+' limited to a maximum of 1.0')
					R_ionic[i,1] = min(R_ionic[i,1],1.0)
		else:
			R_ionic[0,:] = np.array([p_ionic*(1.-range_int),p_ionic*(1.+range_int)])	

		R_land = np.zeros((N_land,2))
		if N_land > 1:
			for i,pp in enumerate(p_land):
				if len(param_range_land)>0:
					if params_name_land[i] in param_range_land[0]:
						for j,pr in enumerate(param_range_land[0]):
							if pr==params_name_land[i]:
								R_land[i,:] = np.array([pp*(1.-param_range_land[1][j]),pp*(1.+param_range_land[1][j])])
					else:
						R_land[i,:] = np.array([pp*(1.-range_int),pp*(1.+range_int)])
				else:
					R_land[i,:] = np.array([pp*(1.-range_int),pp*(1.+range_int)])	
		else:
			R_land[0,:] = np.array([p_land*(1.-range_int),p_land*(1.+range_int)])	

		R = np.concatenate((R_ionic,R_land),axis=0)

		H = lhd(R, N_samples)

		cmd = "mkdir -p "+output_folder+"/param/"+ionic
		os.system(cmd)

		cmd = "mkdir -p "+output_folder+"/param/Land/"
		os.system(cmd)

		for i in range(N_samples):
			if string_header_ionic != '':
				string = string_header_ionic+','
			else:
				string = ''
			for j in range(len(params_name_ionic)-1):
				string += params_name_ionic[j]+'='+str(H[i,j])+','
			string += params_name_ionic[-1]+'='+str(H[i,len(params_name_ionic)-1])
			
			text_file = open(output_folder+'/param/'+ionic+'/'+str(i)+'_param.txt', 'w')
			n = text_file.write(string)
			text_file.close()
		
			if string_header_land != '':
				string = string_header_land+','
			else:
				string = ''
			for j in range(len(params_name_land)-1):
				string += params_name_land[j]+'='+str(H[i,len(params_name_ionic)+j])+','
			string += params_name_land[-1]+'='+str(H[i,len(params_name_ionic)+j+1])
			
			text_file = open(output_folder+'/param/Land/'+str(i)+'_param.txt', 'w')
			n = text_file.write(string)
			text_file.close()

		np.savetxt(output_folder+'/data/X.txt', H, fmt='%.10f')
		np.savetxt(output_folder+'/data/X_'+ionic+'.txt', H[:,:len(params_name_ionic)], fmt='%g')
		np.savetxt(output_folder+'/data/X_land.txt', H[:,len(params_name_ionic):], fmt='%g')

	elif ionic_sample == 'HM':
		print('Sampling Land model only...')

		cmd = "mkdir -p "+output_folder+"/param/Land/"
		os.system(cmd)

		if not os.path.exists(waveload+'/X_'+ionic+'.txt'):
			raise Exception('The waveload folder needs to containg X_ionic.txt contaning ionic model samples')
		H_ionic = np.loadtxt(waveload+'/X_'+ionic+'.txt')

		if H_ionic.shape[0]!=N_samples:
			raise Exception('The ionic samples from the HM need to be '+str(N_samples))

		R_land = np.zeros((len(p_land),2))
		for i,pp in enumerate(p_land):
			R_land[i,:] = np.array([pp*(1.-range_int),pp*(1.+range_int)])

		H_land = lhd(R_land, N_samples)

		H = np.concatenate((H_ionic,H_land),axis=1)

		for i in range(N_samples):
		
			string = ''
			for j in range(len(params_name_land)-1):
				string += params_name_land[j]+'='+str(H_land[i,j])+','
			string += params_name_land[-1]+'='+str(H_land[i,j+1])
			
			text_file = open(output_folder+'/param/Land/'+str(i)+'_param.txt', 'w')
			n = text_file.write(string)
			text_file.close()

		np.savetxt(output_folder+'/data/X_land.txt', H_land, fmt='%g')
		np.savetxt(output_folder+'/data/X.txt', H, fmt='%g')
		
	