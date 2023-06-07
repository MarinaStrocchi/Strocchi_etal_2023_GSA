import os

import numpy as np

import matplotlib.pyplot as plt

from gpytGPE.utils.design import read_labels

from GSA_library.file_utils import read_ionic_output,write_calcium_land

def compute_vm_output(Vm):

	"""
	Computes output from transmembrane potential trace. It assumes it 
	is the last beat already

	Args:
		- Vm: transmembrane potential trace (one beat)

	Outputs:
		- array of outputs: [Vm rest, Vm peak, dVdt_max, APD90]

	"""	

	Vm_rest = Vm[0]
	Vm_peak = np.max(Vm)
	dVmdt_max = -1.

	th = (Vm_peak-Vm_rest)*0.1+Vm_rest
	tmp = np.where(Vm>th)[0]

	if tmp.size:
		APD90 = tmp[-1]-tmp[0]
	else:
		APD90 = -1

	return np.array([Vm_rest,Vm_peak,dVmdt_max,APD90])

def compute_ca_output(Ca_i,
					  ionic='ToRORd'):

	"""
	Computes output from calcium transient trace. It assumes it 
	is the last beat already

	Args:
	    - Ca_i: calcium transient trace (one beat)
	    - ionic: ToRORd or COURTEMANCHE - changes what outputs are computed

	Outputs:
		- array of outputs: [Ca diast, Ca ampl, TTP, RT90] for ToRORd 
						and [Ca diast, dCadt_max, TTP, RT90] for COURTEMANCHE 

	"""	

	Ca_diast = Ca_i[0]
	Ca_ampl = np.max(Ca_i)-Ca_diast

	if ionic == 'ToRORd':
		ind = np.where(Ca_i == np.max(Ca_i))[0]
		TTP = ind[0]
	elif ionic == 'COURTEMANCHE':
		dCa_i = np.gradient(Ca_i)
		dCadt_max = np.max(dCa_i)

	th = Ca_ampl*0.1+Ca_diast
	tmp = np.where(Ca_i>th)[0]
	if tmp.size:
		ind = np.where(Ca_i == np.max(Ca_i))[0]
		TTP = ind[0]
		RT90 = tmp[-1]-TTP
	else:
		RT90 = -1

	if ionic == 'ToRORd':
		output = np.array([Ca_diast,Ca_ampl,TTP,RT90])
	elif ionic == 'COURTEMANCHE':
		output = np.array([Ca_diast,Ca_ampl,dCadt_max,RT90])

	return output

def compute_tension_output(Tension,
						   lambda_out,
						   isometric=False):

	"""
	Computes output from tension transient trace. It assumes it 
	is the last beat already

	Args:
	    - Tension: tension transient trace (one beat)
	    - lambda_out: stretch transient trace if simulation is not isometric
	    - isometric: True if isometric and lambda shortening is not computed

	Outputs:
		- array of outputs: [T_peak,TTP,dTdt_max,dTdt_min,Tdur,T_rest,lambda_c]
							lambda_c is 0 if isometric=True

	"""	

	T_peak = np.max(Tension)

	ind = np.where(Tension==T_peak)[0]
	TTP = ind[0]

	dTdt = np.gradient(Tension)
	dTdt_max = np.max(dTdt)
	dTdt_min = np.min(dTdt)

	T_rest = Tension[0]

	th = (T_peak-T_rest)*0.05+T_rest
	temp_ta = np.where(Tension>=th)[0]
	Tdur = temp_ta[-1] - temp_ta[0]

	if not isometric:
		lambda_c = (lambda_out[0]-np.min(lambda_out))/lambda_out[0]*100
	else:
		lambda_c = 0.0

	return np.array([T_peak,TTP,dTdt_max,dTdt_min,Tdur,T_rest,lambda_c])

def torord_output(baseFolder,
				  sim_foldername='sims/',
				  start_sample=0,
				  last_sample=50,
				  mode='all',
				  land=False,
				  output_file=None):

	"""
	Computes output for all simulations in baseFolder/sims/ i from
	start_sample to last_sample

	Args:
		- baseFolder: folder containing all simulations
		- sim_foldername: baseFolder/sim_foldername/
		- start_sample: first simulation number
		- last_sample: last simulation number
		- mode: all,Vm,Ca_i. If all, both Vm and Ca_i outputs are computed
		- land: if True, Ca_i_last.dat is expected
		output_file: baseFolder/data/Y.txt by default containing all output values

	"""	

	if output_file is None:
		output_file = baseFolder+'/data/Y.txt'

	if mode == 'all':
		output = np.zeros((last_sample-start_sample+1,8))
	else:
		output = np.zeros((last_sample-start_sample+1,4))
	count = 0
	for i in range(start_sample,last_sample+1):

		folder = baseFolder+'/'+sim_foldername+'/'+str(i)

		print('Computing output for ToRORd model '+folder+'...')
		
		if land:
			t = np.arange(0,Ca_i.shape[0])
			write_calcium_land(t,Ca_i,folder+'/Ca_i_land.dat')

		if mode == 'all':
			if not os.path.exists(folder+'/Vm.dat'):
				raise Exception('Cannot find input file. You need to have baseFolder/sim_foldername/i/Vm.dat')

			if not os.path.exists(folder+'/Ca_i.dat'):
				raise Exception('Cannot find input file. You need to have baseFolder/sim_foldername/i/Ca_i.dat')
			
			Vm = read_ionic_output(folder+'/Vm.dat')
			Ca_i = read_ionic_output(folder+'/Ca_i.dat')

			output[i,0:4] = compute_vm_output(Vm)
			output[i,4:8] = compute_ca_output(Ca_i,ionic='ToRORd')
		elif mode == 'Vm':
			if not os.path.exists(folder+'/Vm.dat'):
				raise Exception('Cannot find input file. You need to have baseFolder/sim_foldername/i/Vm.dat')

			Vm = read_ionic_output(folder+'/Vm.dat')
			output[i,0:4] = compute_vm_output(Vm)
		elif mode == 'Ca_i':
			if not os.path.exists(folder+'/Ca_i.dat'):
				raise Exception('Cannot find input file'+folder+'/Ca_i.dat. You need to have baseFolder/sim_foldername/i/Ca_i.dat')

			Ca_i = read_ionic_output(folder+'/Ca_i.dat')

			if Ca_i.size:
				output[i,0:4] = compute_ca_output(Ca_i,ionic='ToRORd')
			else:
				print('Skipping this sample becasue the output is empty.')

		count += 1

	np.savetxt(output_file,output)

def courtemanche_output(baseFolder,
						sim_foldername='sims/',
						start_sample=0,
						last_sample=50,
						mode='all',
						land=False,
						output_file=None):

	"""
	Computes output for all simulations in baseFolder/sims/ i from
	start_sample to last_sample

	Args:
		- baseFolder: folder containing all simulations
		- sim_foldername: baseFolder/sim_foldername/
		- start_sample: first simulation number
		- last_sample: last simulation number
		- mode: all,Vm,Ca_i. If all, both Vm and Ca_i outputs are computed
		- land: if True, Ca_i_last.dat is expected
		output_file: baseFolder/data/Y.txt by default containing all output values

	"""	

	if output_file is None:
		output_file = baseFolder+'/data/Y.txt'
	
	if mode == 'all':
		output = np.zeros((last_sample-start_sample+1,8))
	else:
		output = np.zeros((last_sample-start_sample+1,4))

	count = 0
	for i in range(start_sample,last_sample+1):

		folder = baseFolder+'/'+sim_foldername+'/'+str(i)

		print('Computing output for COURTEMANCHE model '+folder+'...')

		if mode == 'all':
			if not os.path.exists(folder+'/Vm.dat'):
				raise Exception('Cannot find input file. You need to have baseFolder/sim_foldername/i/Vm.dat')

			if not os.path.exists(folder+'/Ca_i.dat'):
				raise Exception('Cannot find input file. You need to have baseFolder/sim_foldername/i/Ca_i.dat')

			Vm = read_ionic_output(folder+'/Vm.dat')
			Ca_i = read_ionic_output(folder+'/Ca_i.dat')
			output[i,0:4] = compute_vm_output(Vm)
			output[i,4:8] = compute_ca_output(Ca_i,ionic='COURTEMANCHE')
		elif mode == 'Vm':
			if not os.path.exists(folder+'/Vm.dat'):
				raise Exception('Cannot find input file. You need to have baseFolder/sim_foldername/i/Vm.dat')

			Vm = read_ionic_output(folder+'/Vm.dat')
			output[i,0:4] = compute_vm_output(Vm)
		elif mode == 'Ca_i':
			if not os.path.exists(folder+'/Ca_i.dat'):
				raise Exception('Cannot find input file. You need to have baseFolder/sim_foldername/i/Ca_i.dat')

			Ca_i = read_ionic_output(folder+'/Ca_i.dat')
			if Ca_i.size:
				output[i,0:4] = compute_ca_output(Ca_i,ionic='COURTEMANCHE')
			else:
				print('Skipping this sample becasue the output is empty.')

		count += 1

	np.savetxt(output_file,output)

def land_output(baseFolder,
				sim_foldername='sims/',
				start_sample=0,
				last_sample=50,
				isometric=False,
				output_file=None):

	"""
	Computes output for all simulations in baseFolder/sims/ i from
	start_sample to last_sample

	Args:
		- baseFolder: folder containing all simulations
		- sim_foldername: baseFolder/sim_foldername/
		- start_sample: first simulation number
		- last_sample: last simulation number
		- isometric: if True, no stretch.dat is required
		output_file: baseFolder/data/Y.txt by default containing all output values

	"""	

	if output_file is None:
		output_file = baseFolder+'/data/Y.txt'

	output = np.zeros((last_sample-start_sample+1,7))
	count = 0
	for i in range(start_sample,last_sample+1):

		folder = baseFolder+'/'+sim_foldername+'/'+str(i)

		print('Computing output for Land model '+folder+'...')

		if not os.path.exists(folder+'/Tension.dat'):
				raise Exception('Cannot find input file. You need to have baseFolder/sim_foldername/i/Tension.dat')

		Tension = read_ionic_output(folder+'/Tension.dat')
		# lambda_out = read_ionic_output(folder+'/lambda.dat')
		if not isometric:
			if not os.path.exists(folder+'/stretch.dat'):
				raise Exception('Cannot find input file. You need to have baseFolder/sim_foldername/i/stretch.dat')
			
			lambda_out = read_ionic_output(folder+'/stretch.dat')
		else:
			lambda_out = None

		if Tension.size and abs(Tension[0]-Tension[-1]<5e-02):
			output[i,:] = compute_tension_output(Tension,lambda_out,isometric)

		count += 1

	np.savetxt(output_file,output)

def filter_ionic_output(baseFolder,
						X_file_name="X.txt",
						Y_file_name="Y.txt",
						X_file_output="X_filt.txt",
						Y_file_output="Y_filt.txt",
						output_bound_file="output_bound.txt"):

	"""
	Filters simulation outputs based on output bounds. Put NaN
	if you don't want to limit the outputs

	Args:
		- baseFolder: folder containing all files
		- X_file_name: input X.txt
		- Y_file_name: input Y.txt
		- X_file_output: filtered X_filt.txt
		- Y_file_output: filtered Y_filt.txt
		- output_bound_file: output bounds [min max] for each

	"""	

	if not os.path.exists(baseFolder+"/"+X_file_name):
		raise Exception('The input X filename does not exist')
	X = np.loadtxt(baseFolder+"/"+X_file_name,dtype=float)

	if not os.path.exists(baseFolder+"/"+Y_file_name):
		raise Exception('The input Y filename does not exist')
	Y = np.loadtxt(baseFolder+"/"+Y_file_name,dtype=float)

	if X.shape[0]!=Y.shape[0]:
		raise Exception('X and Y need to have the same size.')

	B = np.loadtxt(baseFolder+"/"+output_bound_file,dtype=float)
	if B.shape[0]!=Y.shape[1]:
		raise Exception('Bound and Y.txt dimensions do not match.')	

	mask = np.ones((Y.shape[0],),dtype=bool)
	for i in range(Y.shape[1]):
		if not np.isnan(B[i,:]).all():
			if not np.isnan(B[i,0]):
				mask[np.where(Y[:,i]<=B[i,0])[0]] = False
			if not np.isnan(B[i,1]):
				mask[np.where(Y[:,i]>=B[i,1])[0]] = False
	X_filtered = X[mask,:]
	Y_filtered = Y[mask,:]

	np.savetxt(baseFolder+"/"+X_file_output,X_filtered,fmt='%g')

	np.savetxt(baseFolder+"/"+Y_file_output,Y_filtered,fmt='%g')

	np.savetxt(baseFolder+"/output_mask.txt",mask,fmt='%s')
			
def string_to_bool(mask_str):

	"""
	Converts a True/False vector into a numpy
	array with 1/0

	Args:
		- mask_str: list of True/False

	Outputs:
		- mask_bool: numpy array with 1=True, 0=False
	"""	

	mask_bool = np.zeros((len(mask_str),),dtype=bool)
	for i,v in enumerate(mask_str):
		if v == 'True':
			mask_bool[i] = 1
			
	return mask_bool

def apply_output_mask(output_file,mask_file,Y_file_output="Y_filt.txt"):

	"""
	Applies a mask (1=True, 0=False) to an output array.

	Args:
		- output_file: file to be filtered
		- mask_file: file containing the mask
		- Y_file_output: name of the file to output

	"""	

	Y = np.loadtxt(output_file)
	
	mask = read_labels(mask_file)
	mask_bool = string_to_bool(mask)

	Y_filtered = Y[np.where(mask_bool==1)[0],:]

	np.savetxt(Y_file_output,Y_filtered)

def check_failed(baseFolder,
				sim_foldername='sims/',
				start_sample=0,
				last_sample=50):

	"""
	Construct a mask by checking whish simulation failed

	Args:
		- baseFolder: where to find the simulations
		- sim_foldername: subfolder in baseFolder where to find the simulations
		- start_sample: first sample to analyse
		- last_sample: last sample to analyse
		
	Outputs:
		- mask: array with 1=Ok and 0=failed simulation

	"""	

	mask = np.zeros((last_sample+1-start_sample,),dtype=int)
	for i in range(start_sample,last_sample+1):

		folder = baseFolder+'/'+sim_foldername+'/'+str(i)

		if not os.path.exists(folder+'/Tension.dat'):
			raise Exception('Cannot find input file. You need to have baseFolder/sim_foldername/i/Tension.dat')
		Tension = read_ionic_output(folder+'/Tension.dat')

		if not np.isnan(np.abs(Tension[0])):
			mask[i] = 1

	return mask

