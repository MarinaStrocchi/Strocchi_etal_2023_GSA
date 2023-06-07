import os

import json

import numpy as np
from pandas import read_csv

import matplotlib.pyplot as plt

from gpytGPE.utils.design import read_labels

from GSA_library.file_utils import read_json, write_json

def VV_output(vv,
			  nbeats=10,
			  BCL=850):

	"""
	Compute ventricles output from circadapt simulation.

	Args:
		- vv: cavity dictionary from CircAdapt cavity output
		- BCL: cycle length in milliseconds, used to extract last last_beat
		- nbeats: number of simulated beats 

	Outputs:
		- vv_output: array of outputs [EDV, EDP, p @ End IVC, pMax, ESV, p @ end Ejection]

	"""

	start = BCL*(nbeats-1)
	end = BCL*nbeats

	vv_output = np.zeros((6,))

	last_beat = np.where(np.array(vv['Time'])>=start)[0]
	volume = np.array(vv['Volume'][last_beat])
	pressure = np.array(vv['Pressure'][last_beat])
	time = np.array(vv['Time'][last_beat])
	
	dv = np.gradient(volume)
	ind_IVC_ = np.intersect1d(np.where(np.abs(dv)<=0.1)[0],np.where(time<=start+BCL/3)[0])
	jump = np.where(np.gradient(ind_IVC_)>1)[0]

	if len(jump) == 0:
		ind_IVC = ind_IVC_
	else:
		ind_IVC = ind_IVC_[jump[-1]:-1]

	ind_IVR_ = np.intersect1d(np.where(np.abs(dv)<=0.1)[0],np.where(time>=start+BCL/2)[0])
	jump = np.where(np.gradient(ind_IVR_)>1)[0]

	if len(jump) == 0:
		ind_IVR = ind_IVR_
	else:
		ind_IVR = ind_IVR_[0:jump[0]]

	ind_ED = ind_IVC[0]
	ind_endIVC = ind_IVC[-1]
	ind_begIVR = ind_IVR[0]

	vv_output[0] = volume[ind_ED]
	vv_output[1] = pressure[ind_ED]
	vv_output[2] = pressure[ind_endIVC]

	# max P
	ind_maxP = np.where(pressure == np.max(pressure))[0]
	if len(ind_maxP)>0:
		ind_maxP = ind_maxP[0]
	vv_output[3] = np.max(pressure)

	vv_output[5] = pressure[ind_begIVR]

	vv_output[4] = np.min(volume)

	# vv_output[7] = (vv_output[0]-vv_output[5])/vv_output[0]*100

	return vv_output

def AA_output(aa,
			  nbeats=10,
			  BCL=850):

	"""
	Compute atria output from circadapt simulation.

	Args:
		- aa: cavity dictionary from CircAdapt cavity output
		- BCL: cycle length in milliseconds, used to extract last last_beat
		- nbeats: number of simulated beats 

	Outputs:
		- aa_output: array of outputs [EDV_awave,pMax_awave,ESV_awave,ESV_vwave,
									   pMax_vwave,ESV_vwave,pMin_vwave]

	"""

	aa_output = np.zeros((7,))

	start = BCL*(nbeats-1)
	end = BCL*nbeats

	last_beat = np.where(np.array(aa['Time'])>=start)[0]
	volume = np.array(aa['Volume'][last_beat])
	pressure = np.array(aa['Pressure'][last_beat])
	time = np.array(aa['Time'][last_beat])

	time_awave = end-BCL+200
	time_vwave = end-BCL/2
	ind_awave = np.where(np.array(time)<=time_awave)[0]
	ind_vwave = np.where(np.array(time)>=time_vwave)[0]

	v_awave = np.array(volume)[ind_awave]
	p_awave = np.array(pressure)[ind_awave]

	v_vwave = np.array(volume)[ind_vwave]
	p_vwave = np.array(pressure)[ind_vwave]

	aa_output[0] = np.max(v_awave)
	aa_output[1] = np.max(p_awave)
	aa_output[2] = np.min(v_awave)
	# aa_output[7] = (aa_output[0]-aa_output[2])/aa_output[0]*100
	
	aa_output[3] = np.max(v_vwave)
	aa_output[4] = np.max(p_vwave)
	aa_output[5] = np.min(v_vwave)
	aa_output[6] = np.min(p_vwave)

	return aa_output

def write_output(baseFolder,
				 LV_output,
				 RV_output,
				 LA_output,
				 RA_output,
				 vv_keys,
				 aa_keys):

	"""
	Writes Output.json file in the baseFolder for a CircAdapt simulation

	Args:
		- baseFolder: output folder
		- LV_output: array of ventricular outputs for the LV
		- RV_output: array of ventricular outputs for the RV
		- LA_output: array of atria outputs for the LA
		- RA_output: array of atria outputs for the RA
		- vv_keys: labels for the ventricular outputs
		- aa_keys: labels for the atroa outputs

	"""

	output = {'LV':{}, 'RV':{}, 'LA':{}, 'RA':{}}
	
	for i in range(len(vv_keys)):
		output['LV'][vv_keys[i]] = round(LV_output[i],2)
		output['RV'][vv_keys[i]] = round(RV_output[i],2)

	for i in range(len(aa_keys)):
		output['LA'][aa_keys[i]] = round(LA_output[i],2)
		output['RA'][aa_keys[i]] = round(RA_output[i],2)

	write_json(output,baseFolder+'/Output.json')


def computOutput(baseFolder,
				 start_sample=1,
				 last_sample=10,
				 nbeats=10,
				 BCL=850):

	"""
	Computes CircAdapt output for simulations in baseFolder/sims/

	Args:
		- baseFolder: contains all simulations
		- start_sample: number of the first simulation
		- last_sample: number of the last simulation
		- nbeats: number of simulated beats
		- BCL: basic cycle length in milliseconds

	Outputs:
		Saves a LV,RV,LA,RA_output.txt and Y.txt files in
		baseFolder/data/ containing the computed output.
		The outputs are all 0. if the simulation failed

	"""

	LV_output = np.zeros((last_sample-start_sample+1,6))
	LA_output = np.zeros((last_sample-start_sample+1,7))
	RV_output = np.zeros((last_sample-start_sample+1,6))
	RA_output = np.zeros((last_sample-start_sample+1,7))

	if not os.path.exists(baseFolder+'/data/ylabels_v.txt'):
		raise Exception('You need to define baseFolder/data/ylabels_v.txt containing labels for ventricular outputs')
	if not os.path.exists(baseFolder+'/data/ylabels_a.txt'):
		raise Exception('You need to define baseFolder/data/ylabels_a.txt containing labels for atria outputs')
	
	vv_keys = read_labels(baseFolder+'/data/ylabels_v.txt')
	aa_keys = read_labels(baseFolder+'/data/ylabels_a.txt')

	count = 0
	for i in range(start_sample,last_sample+1):
		print('Simulation '+str(i)+'...')
		folder = baseFolder+'sims/'+str(i)

		if not os.path.exists(folder+'/cav.LV.csv'):
			raise Exception('Cannot read output file. The folder structure needs to be baseFolder/sims/i/cav.LV,RV,LA,RA.csv')

		lv = read_csv(folder+'/cav.LV.csv', delimiter=",", skipinitialspace=True,
		   header=0, comment='#')
		rv = read_csv(folder+'/cav.RV.csv', delimiter=",", skipinitialspace=True,
		   header=0, comment='#')
		la = read_csv(folder+'/cav.LA.csv', delimiter=",", skipinitialspace=True,
		   header=0, comment='#')
		ra = read_csv(folder+'/cav.RA.csv', delimiter=",", skipinitialspace=True,
		   header=0, comment='#')
		# compute LV metrics
		if len(lv)>0 and max(lv['Time']) == BCL*nbeats:
			LV_output[count,:] = VV_output(lv,nbeats=nbeats,BCL=BCL)
			RV_output[count,:] = VV_output(rv,nbeats=nbeats,BCL=BCL)
			LA_output[count,:] = AA_output(la,nbeats=nbeats,BCL=BCL)
			RA_output[count,:] = AA_output(ra,nbeats=nbeats,BCL=BCL)

		write_output(folder,LV_output[count,:],RV_output[count,:],LA_output[count,:],RA_output[count,:],
			vv_keys,aa_keys)
		count += 1
	
	np.savetxt(baseFolder+'data/LV_output.txt', LV_output, fmt='%.2f')	
	np.savetxt(baseFolder+'data/RV_output.txt', RV_output, fmt='%.2f')	
	np.savetxt(baseFolder+'data/LA_output.txt', LA_output, fmt='%.2f')	
	np.savetxt(baseFolder+'data/RA_output.txt', RA_output, fmt='%.2f')	
	np.savetxt(baseFolder+'data/Y.txt', np.concatenate((LV_output,RV_output,LA_output,RA_output),axis=1), fmt='%.2f')	

def filter_output(dataFolder):

	"""
	Filter simulation outputs based on values. If all 0. then the simulation
	is considered to fail

	Args:
		- dataFolder: contains file Y.txt containing all simulations outputs.
					  It modifies X.txt as well to get rid of failed parameter
					  combinations

	Outputs:
		Outputs: saves X_filt.txt, Y_filt.txt and output_mask.txt in dataFolder
		containing the filtered inputs, outputs and mask to get them from the
		initial dataset.

	"""

	if not os.path.exists(dataFolder + "X.txt") or not not os.path.exists(dataFolder + "Y.txt"):
		raise Exception('Cannot read X.txt or Y.txt from dataFolder.')

	X=np.loadtxt(dataFolder + "X.txt", dtype=float)
	Y=np.loadtxt(dataFolder + "Y.txt", dtype=float)

	ind_ok = np.where(np.sum(Y,axis=1)!=0)[0]

	mask = np.zeros((Y.shape[0],),dtype=bool)
	mask[ind_ok] = 1

	np.savetxt(dataFolder+'/X_filt.txt',X[ind_ok,:])
	np.savetxt(dataFolder+'/Y_filt.txt',Y[ind_ok,:])
	np.savetxt(dataFolder+"/output_mask.txt",mask,fmt='%s')
