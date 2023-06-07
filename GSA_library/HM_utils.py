import os

import math

import numpy as np
from scipy.spatial import ConvexHull, convex_hull_plot_2d
from shapely.geometry import Polygon

import matplotlib.pyplot as plt
import matplotlib.patches as patch

from Historia.history import hm

from gpytGPE.gpe import GPEmul
from gpytGPE.utils.design import read_labels

def combine_datasets(datafolder1,filename1,datafolder2,filename2,outputfile):

	"""
	Concatenates two datasets 

	Args: 
		- datafolder1: where the first dataset is
		- filename1: what the first dataset is called
		- datafolder2: where the second dataset is
		- filename2: what the second dataset is called
		- outputfile: where to save the combined datasets

	"""

	M1 = np.loadtxt(datafolder1+'/'+filename1)
	M2 = np.loadtxt(datafolder2+'/'+filename2)

	M = np.concatenate((M1,M2))

	np.savetxt(outputfile,M)

def combine_exp_datasets(datafolder1,datafolder2,N1,N2,outputfolder):

	"""
	Combine two experimental datasets given their means and standard 
	deviations

	Args: 
		- datafolder1: where the first dataset is (should contain mean.txt and std.txt)
		- datafolder2: where the second dataset is (should contain mean.txt and std.txt)
		- N1: how many samples the first dataset has
		- N2: how many samples the second dataset has
		- outputfolder: where to save the combined dataset
		
	"""

	cmd="mkdir "+outputfolder
	os.system(cmd)

	m1 = np.loadtxt(datafolder1+'/mean.txt')
	m2 = np.loadtxt(datafolder2+'/mean.txt')

	s1 = np.loadtxt(datafolder1+'/std.txt')
	s2 = np.loadtxt(datafolder2+'/std.txt')

	m1_nan = np.where(np.isnan(m1))[0]
	m2_nan = np.where(np.isnan(m2))[0]
	
	m = (N1*m1+N2*m2)/(N1+N2)
	# s = np.sqrt(((N1-1)*np.power(s1,2)+(N2-2)*np.power(s2,2)
	# 	+N1*np.power((m1-m),2)+N2*np.power((m2-m),2))/(N1+N2-1))

	s = np.sqrt(((N1-1)*np.power(s1,2)+(N2-1)*np.power(s2,2)
		+N1*np.power((m1-m),2)+N2*np.power((m2-m),2))/(N1+N2-1))

	if m1_nan.size != 0:
		m[m1_nan] = m2[m1_nan]
		s[m1_nan] = s2[m1_nan]

	if m2_nan.size != 0:
		m[m2_nan] = m1[m2_nan]
		s[m2_nan] = s1[m2_nan]

	np.savetxt(outputfolder+'exp_mean.txt',m)
	np.savetxt(outputfolder+'exp_std.txt',s)

def combine_exp_datasets_range(datafolder1,datafolder2,outputfolder,std_range=2.0):

	"""
	Combine two experimental datasets given their means and standard 
	deviations, by taking the experimental range as mean+/-std_range*std

	Args: 
		- datafolder1: where the first dataset is (should contain mean.txt and std.txt)
		- datafolder2: where the second dataset is (should contain mean.txt and std.txt)
		- outputfolder: where to save the combined dataset
		- std_range: how many standard deviations away from the mean we set the range
		
	"""

	m1 = np.loadtxt(datafolder1+'/mean.txt')
	m2 = np.loadtxt(datafolder2+'/mean.txt')

	s1 = np.loadtxt(datafolder1+'/std.txt')
	s2 = np.loadtxt(datafolder2+'/std.txt')

	data_min = np.zeros((m1.shape[0],2)) 
	data_max = np.zeros((m1.shape[0],2)) 
	data_min[:,0] = m1-s1*std_range
	data_min[:,1] = m2-s2*std_range
	data_max[:,0] = m1+s1*std_range
	data_max[:,1] = m2+s2*std_range

	min_range = np.min(data_min,axis=1)
	max_range = np.max(data_max,axis=1)

	m = 0.5*(min_range+max_range)
	s = 0.5*(max_range-min_range)

	m1_nan = np.where(np.isnan(m1))[0]
	m2_nan = np.where(np.isnan(m2))[0]
	
	if m1_nan.size != 0:
		m[m1_nan] = m2[m1_nan]
		s[m1_nan] = s2[m1_nan]*std_range

	if m2_nan.size != 0:
		m[m2_nan] = m1[m2_nan]
		s[m2_nan] = s1[m2_nan]*std_range

	np.savetxt(outputfolder+'exp_mean.txt',m)
	np.savetxt(outputfolder+'exp_std.txt',s)

def find_final_interval(NIMP):

	"""
	Tries to find the convex hull containing the non-implausible
	area of a history matching wave - FAILED 

	Args: 
		- NIMP: non-implausible samples

	"""

	hull = ConvexHull(NIMP)
	points = NIMP[hull.vertices,:]

	points_poly = []
	for p in points:
		points_poly.append((p[0],p[1]))

	poly_NIMP = Polygon(points_poly) 

	fig,axis = plt.subplots()
	plt.plot(points[:,0],points[:,1], 'r--', lw=2)
	plt.scatter(NIMP[:,0],NIMP[:,1])

	m = np.mean(NIMP,axis=0)
	s = np.std(NIMP,axis=0)

	k = 0.1
	increment = 0.05
	rect = Polygon([(m[0]-s[0]*k,m[1]-s[1]*k), 
					(m[0]+s[0]*k,m[1]-s[1]*k),
					(m[0]+s[0]*k,m[1]+s[1]*k),
					(m[0]-s[0]*k,m[1]+s[1]*k)])

	while(poly_NIMP.contains(rect)):
		k_old = k
		k += increment
		p0 = (m[0]-s[0]*k,m[1]-s[1]*k)
		p1 = (m[0]+s[0]*k,m[1]-s[1]*k)
		p2 = (m[0]+s[0]*k,m[1]+s[1]*k)
		p3 = (m[0]-s[0]*k,m[1]+s[1]*k)
		rect = Polygon([p0,p1,p2,p3])

	axis.scatter(m[0],m[1],marker='d',s=30,ec='black',fc='black')
	rect_plot = patch.Rectangle(p0,2*k*s[0],2*k*s[1],
					 facecolor='none',
					 linestyle='-',
					 linewidth=2.0,
					 edgecolor='black')

	axis.add_patch(rect_plot)

	plt.show()

	return k_old

def find_final_interval_2D(NIMP):

	"""
	Tries to find the convex hull containing the non-implausible
	area of a history matching wave in 2D - FAILED 

	Args: 
		- NIMP: non-implausible samples

	"""

	hull = ConvexHull(NIMP)
	points = NIMP[hull.vertices,:]

	points_poly = []
	for p in points:
		points_poly.append((p[0],p[1]))

	poly_NIMP = Polygon(points_poly) 

	fig,axis = plt.subplots()
	plt.plot(points[:,0],points[:,1], 'r--', lw=2)
	plt.scatter(NIMP[:,0],NIMP[:,1])

	m = np.mean(NIMP,axis=0)
	s = np.std(NIMP,axis=0)

	increment = 0.05
	krange = np.arange(0.1,3.0,increment)

	area = 0.0
	for k1 in krange:
		for k2 in krange:
			p0 = (m[0]-s[0]*k1,m[1]-s[1]*k2)
			p1 = (m[0]+s[0]*k1,m[1]-s[1]*k2)
			p2 = (m[0]+s[0]*k1,m[1]+s[1]*k2)
			p3 = (m[0]-s[0]*k1,m[1]+s[1]*k2)
			rect = Polygon([p0,p1,p2,p3])

			if poly_NIMP.contains(rect):
				area_new = 2*k1*s[0]*2*k2*s[1]
				if area_new>area:
					area = area_new
					k_store = [k1,k2]


	axis.scatter(m[0],m[1],marker='d',s=30,ec='black',fc='black')
	rect_plot = patch.Rectangle([m[0]-s[0]*k_store[0],m[1]-s[1]*k_store[1]],
					 2*k_store[0]*s[0],2*k_store[1]*s[1],
					 facecolor='none',
					 linestyle='-',
					 linewidth=2.0,
					 edgecolor='black')

	axis.add_patch(rect_plot)

	plt.show()

	return k_store

def get_HM_stats(waves,
				 waves_folder,
				 output_file):

	"""
	Computes the stats for a series of history matching waves

	Args: 
		- waves: list of waves to look at [1,2,3,4,5,...]
		- waves_folder: where the waves are
		- output_file: where to save the stats

	"""

	f = open(output_file, "w")
	f.write('wave threshold %NIMP meanI maxI meanV maxV\n')
	for w in waves:
		W = hm.Wave()
		W.load(waves_folder+'wave'+str(w)+'/wave_'+str(w))

		N_nimp = W.nimp_idx.shape[0]
		N = W.n_samples

		NIMP_pp = round(N_nimp/N*100,1)

		I_mean = round(np.mean(W.I),1)
		I_max = round(np.max(W.I),1)
		V_mean = round(np.mean(W.PV),2)
		V_max = round(np.max(W.PV),2)

		out = [W.cutoff,NIMP_pp,I_mean,I_max,V_mean,V_max]
		out = [str(o) for o in out]
		out = ['wave'+str(w),]+out
		f.write(' '.join(out))

		f.write('\n')

	f.close()

def get_HM_stats_outputs(waves_list,
				 		 gpes_folder_list,
				 		 features_file,
				 		 ylabels_file,
				 		 output_file):

	"""
	Computes the stats for a series of history matching waves
	separating output by output

	Args: 
		- waves: list of wave files (with full path) 
		- gpes_folder_list: list of the folder where the GPEs are for each wave
		- features_file: list of features to output
		- output_file: where to save the stats

	"""

	if len(waves_list)!=len(gpes_folder_list):
		raise Exception("You need to provide a GPE folder for each wave. Lengths of lists do not match.")

	if not os.path.exists(features_file):
		raise Exception("Cannot find features file you provided.")
	active_idx = list(np.loadtxt(features_file,dtype=int))

	ylabels = read_labels(ylabels_file)
	ylabels = [ylabels[idx] for idx in active_idx]

	data = np.zeros((len(ylabels)+3,len(waves_list)*2),dtype=float)

	for i in range(len(waves_list)):

		W = hm.Wave()
		W.load(waves_list[i])

		emulator = []
		for idx in active_idx:

			loadpath = gpes_folder_list[i] + str(idx) + "/"
			X_train = np.loadtxt(loadpath + "X_train.txt", dtype=np.float64)
			y_train = np.loadtxt(loadpath + "y_train.txt", dtype=np.float64)
			emul = GPEmul.load( X_train, y_train, loadpath=loadpath ) 
			emulator.append(emul)

		W.emulator = emulator
		W.output_dim = len(emulator)
		W.mean = W.mean[:len(ylabels)]
		W.var = W.var[:len(ylabels)]

		X = np.row_stack((W.NIMP,W.IMP))

		I,PV = W.compute_impl_array(X)

		mean_I = np.mean(I,axis=0)
		mean_PV = np.mean(PV,axis=0)

		for j in range(W.n_samples):
			W.I[j] = np.max(I[j,:])
			W.PV[j] = np.max(PV[j,:])

		data[:-3,i*2] = np.round(mean_I,2)
		data[:-3,2*i+1] = np.round(mean_PV,2)

		data[-3,i*2] = np.round(np.mean(W.I),2)
		data[-3,2*i+1] = np.round(np.mean(W.PV),2)
 		
		data[-2,i*2] = np.round(np.max(W.I),2)
		data[-2,2*i+1] = np.round(np.max(W.PV),2)
 		
		N_nimp = W.nimp_idx.shape[0]
		N = W.n_samples

		NIMP_pp = np.round(N_nimp/N*100,2)
		data[-1,i*2] = NIMP_pp

	print(data)

	header = ['',]+['\multicolumn{2}{|c|}{wave'+str(i)+'}' for i in range(1,len(waves_list)+1)]
	header = '\t&\t'.join(header)+'\t \\ \n'

	f = open(output_file, "w")
	f.write(header)

	header_2 = ['',]+['I','V']*len(waves_list)
	header_2 = '\t&\t'.join(header_2)+'\t \\ \n'
	f.write(header_2)

	for i in range(len(ylabels)):
		f.write(ylabels[i]+'\t&\t')

		data_txt = [str(data[i,j]) for j in range(data.shape[1])]
		data_txt = '\t&\t'.join(data_txt)+'\t \\ \n'

		f.write(data_txt)

	f.write('Overall Mean\t&\t')
	overall = list(data[-3,:])
	overall_txt = [str(o) for o in overall]
	overall_txt = '\t&\t'.join(overall_txt)+'\t \\ \n'

	f.write(overall_txt)

	f.write('Overall Max\t&\t')
	overall = list(data[-2,:])
	overall_txt = [str(o) for o in overall]
	overall_txt = '\t&\t'.join(overall_txt)+'\t \\ \n'

	f.write(overall_txt)

	f.write('NIMP [\%] \t&\t')
	nimp = list(data[-1,:])
	nimp_txt = [str(n) for n in nimp]
	nimp_txt = ['\multicolumn{2}{|c|}{'+n+'\%}' for n in nimp_txt[::2]]
	nimp_txt = '\t&\t'.join(nimp_txt)+'\t \\ \n'

	f.write(nimp_txt)
	f.close()
