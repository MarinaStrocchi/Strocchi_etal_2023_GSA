import os

import json
import re

import numpy as np
import pandas as pd

from reportlab.lib.pagesizes import A4
from reportlab.platypus import SimpleDocTemplate, Table, TableStyle
from reportlab.lib import colors

from gpytGPE.utils.design import read_labels

class MyEncoder(json.JSONEncoder):
    def default(self, obj):
        if isinstance(obj, np.integer):
            return int(obj)
        elif isinstance(obj, np.floating):
            return float(obj)
        elif isinstance(obj, np.ndarray):
            return obj.tolist()
        return json.JSONEncoder.default(self, obj)

def X_to_param(X,
			   params_name,
			   outputpath,
			   string_header='flags=ENDO',
			   adapt_beta1=False,
			   beta_1_default=-2.4,
			   ca50_default=0.805):

	"""
	Converts a matrix X to a series of string (#_param.txt) for a 
	cell model.

	Args:
		X: matrix of the parameter combinations
		params_name: labels for the parameters
		outputpath: where to save the parameter files
		string_header: if you want to have a header for your parameter
					   string. Useful if you want to fix some parameters
					   to a value that is not the default one
		adapt_beta1: beta_1 for length dependence in the Land model can be adapted
					 depending on ca50
		beta_1_default: default value to scale beta_1
		ca50_default: default value for ca50 to scale beta_1

	"""

	if adapt_beta1:

		print('---------------------------------------')
		print('Computing scaled beta_1 values...')
		print('Default beta_1 = '+str(beta_1_default))
		print('Default ca50 = '+str(ca50_default))
		print('---------------------------------------')

		ca50_idx=None
		for k,l in enumerate(params_name):
			if l == 'ca50':
				ca50_idx = k

		if ca50_idx is None:
			raise Exception('Beta_1 can be adapted only based on ca50, and it looks like ca50 is not changing in your simulation.')

		beta_1_scaled = np.zeros((X.shape[0],),dtype=float)
		for j in range(X.shape[0]):
			beta_1_scaled[j] = beta_1_default/ca50_default*X[j,ca50_idx]

	# N = np.size(X[:,0])
	if len(X.shape)>1:
		N = X.shape[0]
	elif (len(X.shape) == 1) and len(params_name)>1:
		N = 1
		X = X.reshape((N,X.shape[0]))
	elif (len(X.shape) == 1) and len(params_name)==1:
		N = X.shape[0]
		X = X.reshape((N,1))

	if X.shape[1] != len(params_name):
		raise Exception('The matrix and the parameters name do not match in size.')

	cmd = 'mkdir -p '+outputpath
	os.system(cmd)

	for i in range(N):
		
		if string_header != '':
			string = string_header+','
		else:
			string = ''

		if adapt_beta1:
			string = string_header+',beta_1='+str(beta_1_scaled[i])+','

		for j in range(len(params_name)-1):
			string += params_name[j]+'='+str(X[i,j])+','
		string += params_name[-1]+'='+str(X[i,-1])
		
		text_file = open(outputpath+'/'+str(i)+'_param.txt', 'w')
		n = text_file.write(string)
		text_file.close()

def read_json(filename):

	"""
	Reads a json file.

	Args:
		- filename: name of json file with json extension

	Outputs:
		- param: json file content 
	"""

	if not os.path.exists(filename):
		raise Exception('Cannot find'+filename+' file. Filename needs to contain .json extension.')

	with open(filename) as json_file:
		param = json.load(json_file)
	return param

def write_json(data,filename):

	"""
	Writes a json file.

	Args:
		- data: dictionary containing json file content
		- filename: .json file with .json extension at the end
	"""

	with open(filename,'w') as json_file:
		json.dump(data, json_file, indent='    ',cls=MyEncoder)

def read_ionic_output(filename):

	"""
	Reads output trace from cell model in CARP, structured
	as  (99000): 0.115298,
		(99001): 0.115257,
		(99002): 0.115352,
		(99003): 0.118156,
		(99004): 0.138681,
	where the number in () is the time and the value after is 
	the trace value

	Args:
		- filename: file to be read in

	Outputs:
		- data: array containing the trace
	"""

	if not os.path.exists(filename):
		raise Exception('Cannot find '+filename+'.')

	f = open(filename)
	lines = f.read().splitlines()

	data = []
	for i,line in enumerate(lines):
		sep = re.split(': |,|\n', line)

		if i==len(lines)-1:
			data.append(float(sep[-1]))
		elif i>0:
			data.append(float(sep[-2]))

	return np.array(data)

def write_calcium_land(t,Ca_i,filename):

	"""
	Writes a calcium transient and makes it periodic 
	by adding the first value at the end of the trace

	Args:
		- t: time for the trace
		- Ca_i: calcium transient trace
		- filename: file where to save the trace
	"""

	if t.shape[0]!=Ca_i.shape[0]:
		raise Exception('Time trace and calcium trace dimensions do not match.')

	with open(filename,'w') as f:
		f.write(f'{t.shape[0]+1}\n')
		for i in range(t.shape[0]):
			f.write(f'{t[i]}\t{Ca_i[i]}\n')
		f.write(f'{t[-1]+1}\t{Ca_i[0]}\n')
	f.close()

def R2score2table(dataFolder,GPEfolder,features_idx_file=None):

	"""
	Writes pdf file to summarise GPE performance.

	Args:
		- dataFolder: folder containing the data used for GPE training
		- GPEfolder: output folder containing trained GPE
	"""	

	print('Checking folder structure...')
	if features_idx_file is None:
		features_idx_file = dataFolder+"/features_idx_list.txt"
	to_check = [features_idx_file,
				dataFolder+"/ylabels.txt"]
	for f in to_check:
		if not os.path.exists(f):
			raise Exception('Cannot find file '+f)
		else:
			print(f+' found.')

	idx_feature = list(np.loadtxt(features_idx_file,dtype=int))
	ylabels = read_labels(dataFolder+"/ylabels.txt")

	document = SimpleDocTemplate(GPEfolder+"/R2Score.pdf", pagesize=A4, title='R2 Scores')
	data = []
	items = []
	header = ['fold-1','fold-2','fold-3','fold-4','fold-5','mean','best']
	data.append(['Output']+header)
	R2_tab = []

	for idx in idx_feature:
		if os.path.isfile(GPEfolder+"/"+str(idx)+"/R2Score_cv.txt"):
			R2 = np.loadtxt(GPEfolder+"/"+str(idx)+"/R2Score_cv.txt")
			R2_tab.append(list(np.round(R2,4))+[round(np.mean(R2),4)]+[round(np.max(R2),4)])
			data.append([ylabels[idx]]+list(R2)+[round(np.mean(R2),4)]+[round(np.max(R2),4)])
		else:
			raise Exception('Cannot find '+GPEfolder+"/"+str(idx)+"/R2Score_cv.txt")
			
	ylabels = [ylabels[idx] for idx in idx_feature]

	df = pd.DataFrame(R2_tab,columns=header,index=ylabels)
	with open(GPEfolder+"/R2Score.tex",'w') as tf:
		tf.write(df.to_latex())

	tab = Table(data)

	tab.setStyle(TableStyle([('INNERGRID', (0,0), (-1,-1), 0.25, colors.black),
	('BOX', (0,0), (-1,-1), 0.25, colors.black),
	('BACKGROUND',(0,0),(0,-1),colors.lightgrey),
	('BACKGROUND',(0,0),(-1,0),colors.lightgrey)
	]))

	# tab.setStyle(TableStyle([('BACKGROUND',(2,1),(2,1),colors.cadetblue)]))

	for ii in range(1,len(data)):
		for jj in range(1,len(data[ii])-2):
			if data[ii][jj]<0.8:
				# tab.setStyle(TableStyle([('BACKGROUND',(ii,jj),(ii,jj+1),colors.cadetblue)]))
				tab.setStyle(TableStyle([('BACKGROUND',(jj,ii),(jj,ii),colors.cadetblue)]))

	items.append(tab)
	document.build(items)


def ISEscore2table(dataFolder,GPEfolder,features_idx_file=None):

	"""
	Writes pdf file to summarise GPE performance.

	Args:
		- dataFolder: folder containing the data used for GPE training
		- GPEfolder: output folder containing trained GPE
	"""	

	print('Checking folder structure...')
	if features_idx_file is None:
		features_idx_file = dataFolder+"/features_idx_list.txt"
	to_check = [features_idx_file,
				dataFolder+"/ylabels.txt"]
	for f in to_check:
		if not os.path.exists(f):
			raise Exception('Cannot find file '+f)
		else:
			print(f+' found.')

	idx_feature = list(np.loadtxt(features_idx_file,dtype=int))
	ylabels = read_labels(dataFolder+"/ylabels.txt")

	document = SimpleDocTemplate(GPEfolder+"/ISEScore.pdf", pagesize=A4, title='ISE Scores')
	data = []
	items = []
	header = ['fold-1','fold-2','fold-3','fold-4','fold-5','mean','best']
	data.append(['Output']+header)
	ISE_tab = []

	for idx in idx_feature:
		if os.path.isfile(GPEfolder+"/"+str(idx)+"/ISE_cv.txt"):
			ISE = np.loadtxt(GPEfolder+"/"+str(idx)+"/ISE_cv.txt")
			ISE_tab.append(list(np.round(ISE,2))+[round(np.mean(ISE),2)]+[round(np.max(ISE),2)])
			data.append([ylabels[idx]]+list(ISE)+[round(np.mean(ISE),2)]+[round(np.max(ISE),2)])
		else:
			raise Exception('Cannot find '+GPEfolder+"/"+str(idx)+"/ISE_cv.txt")
			
	ylabels = [ylabels[idx] for idx in idx_feature]

	df = pd.DataFrame(ISE_tab,columns=header,index=ylabels)
	with open(GPEfolder+"/ISEScore.tex",'w') as tf:
		tf.write(df.to_latex())

	tab = Table(data)

	tab.setStyle(TableStyle([('INNERGRID', (0,0), (-1,-1), 0.25, colors.black),
	('BOX', (0,0), (-1,-1), 0.25, colors.black),
	('BACKGROUND',(0,0),(0,-1),colors.lightgrey),
	('BACKGROUND',(0,0),(-1,0),colors.lightgrey)
	]))

	# tab.setStyle(TableStyle([('BACKGROUND',(2,1),(2,1),colors.cadetblue)]))

	for ii in range(1,len(data)):
		for jj in range(1,len(data[ii])-2):
			if data[ii][jj]<0.8:
				# tab.setStyle(TableStyle([('BACKGROUND',(ii,jj),(ii,jj+1),colors.cadetblue)]))
				tab.setStyle(TableStyle([('BACKGROUND',(jj,ii),(jj,ii),colors.cadetblue)]))

	items.append(tab)
	document.build(items)

def R2_ISEscore2table_paper(dataFolder,GPEfolder,features_idx_file=None):

	"""
	Writes pdf file to summarise GPE performance.

	Args:
		- dataFolder: folder containing the data used for GPE training
		- GPEfolder: output folder containing trained GPE
	"""	

	print('Checking folder structure...')
	if features_idx_file is None:
		features_idx_file = dataFolder+"/features_idx_list.txt"
	to_check = [features_idx_file,
				dataFolder+"/ylabels.txt"]
	for f in to_check:
		if not os.path.exists(f):
			raise Exception('Cannot find file '+f)
		else:
			print(f+' found.')

	idx_feature = list(np.loadtxt(features_idx_file,dtype=int))
	if os.path.exists(dataFolder+"/ylabels_latex.txt"):
		ylabels = read_labels(dataFolder+"/ylabels_latex.txt")
	else:
		ylabels = read_labels(dataFolder+"/ylabels.txt")

	header = ['Meaning','Metric','fold-1','fold-2','fold-3','fold-4','fold-5','mean']
	tab_data = []
	ylabels_tab = []
	for idx in idx_feature:
		ylabels_tab.append(ylabels[idx])
		ylabels_tab.append(' ')

		if os.path.isfile(GPEfolder+"/"+str(idx)+"/R2Score_cv.txt"):
			R2 = np.loadtxt(GPEfolder+"/"+str(idx)+"/R2Score_cv.txt")
			tab_data.append(['Add output meaning','$R^2$']+list(np.round(R2,4))+[round(np.mean(R2),4)])
		else:
			raise Exception('Cannot find '+GPEfolder+"/"+str(idx)+"/ISE_cv.txt")

		if os.path.isfile(GPEfolder+"/"+str(idx)+"/ISE_cv.txt"):
			ISE = np.loadtxt(GPEfolder+"/"+str(idx)+"/ISE_cv.txt")
			tab_data.append([' ','ISE']+list(np.round(ISE,2))+[round(np.mean(ISE),2)])
		else:
			raise Exception('Cannot find '+GPEfolder+"/"+str(idx)+"/ISE_cv.txt")
			
	ylabels = [ylabels[idx] for idx in idx_feature]

	df = pd.DataFrame(tab_data,columns=header,index=ylabels_tab)
	with open(GPEfolder+"/R2_ISEScore.tex",'w') as tf:
		tf.write(df.to_latex())
