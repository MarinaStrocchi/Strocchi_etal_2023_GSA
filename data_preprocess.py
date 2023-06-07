import os
import sys

import numpy as np
import pandas as pd

in_folder = "./example/ToRORd_ionic/"
out_folder = "./example/data/"

if os.path.exists(out_folder):
	print('The output folder already exists. Removing it.')
	os.system("rm -r "+out_folder)

os.mkdir(out_folder)

par = pd.read_csv(in_folder+"/parameters.csv")
outputs = pd.read_csv(in_folder+"/outputs.csv")

print('----------------------------------------------------')
print('Extracting data from '+in_folder+'...')

x_labels = list(par.columns)
X = par.to_numpy(copy=True)

y_labels = list(outputs.columns)
Y = outputs.to_numpy(copy=True)

print('Saving to '+out_folder+'...')

with open(out_folder+'/xlabels.txt', 'a') as f:
	for xl in x_labels:
		f.write(xl+"\n")

with open(out_folder+'/ylabels.txt', 'a') as f:
	for yl in y_labels:
		f.write(yl+"\n")

np.savetxt(out_folder+"/X.txt",X,fmt="%g")
np.savetxt(out_folder+"/Y.txt",Y,fmt="%g")

features_idx = np.arange(Y.shape[1])
np.savetxt(out_folder+"/features_idx_list.txt",features_idx,fmt="%d")

print('Done.')

print('----------------------------------------------------')
