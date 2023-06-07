# Cell to Whole Organ Global Sensitivity Analysis on a Four-chamber Electromechanics Model Using Gaussian Processes Emulators (PLOS Comp Bio)
This repository contains the code for Strocchi et al 2023, Cell to Whole Organ Global Sensitivity Analysis on a Four-chamber Electromechanics
Model Using Gaussian Processes Emulators, under review for publication on PLOS Computational Biology. 

This code depends on the repositories for Gaussian Processes Emulators ([GPEs](https://github.com/stelong/gpytGPE)) and History Matching ([HM](https://github.com/stelong/Historia)) developed by Stefano Longobardi. 

# Installation
All dependencies are taken care of by the pyproject.toml file. Creating a virtual environment is optional but advisable. To create a virtual environment, follow these command line instructions:

````
conda create -n py38 python=3.8.13
conda activate
````

This will create and activate a virtual environment called py38 (or whatever you like) with python 3.8.13. The python3 version is important so do not change it.

You can then clone the reporistory in a folder of your choosing:

````
git clone https://github.com/MarinaStrocchi/Strocchi_etal_2023_GSA.git
````

To install:

````
pip install --upgrade pip
pip install .
````

This should install all packages that the library needs.

# Example

The example shows how to run train Gaussina processes emulators for scalar outputs, run a global sensitivity analysis using the emulators and finally rank the parameters accourding to their sensitivity indices. The example needs a datasets made of two .csv files:

- parameters.csv: input parameters files. The first row contains the parameter labels, while the following rows contain a sample per parameter.
- outputs.csv: output features computed with the ToR-ORd ionic model (https://elifesciences.org/articles/48890). The first row contains the output labels, while the following rows contain a value per output. The rows correspond to the outputs computed with the parameter values in the parameters.csv file.

These two files are provided with the example but they can also be downloaded from the Zenodo database linked to the publication (DOI: 10.5281/zenodo.7405335).

Once you have successfully installed the library, first you need to transform the .csv files into the format the emulators need to be trained. To do this, run:
 ````
 python data_preprocess.py
 ````
 This will create a folder called 'data' in the example folder with the files the emulators need.
 
 To train the emulators, run a sensitivity analysis using the emulators and rank the parameters, run:
  ````
 python example.py
 ````
 
 For each output feature (4 in this case), we train five separate emulators to perform a 5-fold cross validation, to check the performance of the emualtors. We also train an emulator using the whole dataset. This will be used in the sensitivity analysis. At lines 14 of example.py, you can choose if you want to train all 5 emulators for the cross validation in parallel. If set to False, the training might take a while. 
 
 At line 15 of example.py, you can also choose to consider the uncertainties of the emulators in the sensitivity analysis by sampling N=1000 times the posterior distribution of the emulators, and to compute the sensitivity indices for each of these 1000 samples. If you set UNCERTAINTY=True, the sensitivity analysis will take time. Otherwise, if you set UNCERTAINTY=False, the sensitivity analysis will be very quick.
 
 The code will also plot a heatmap with the resulting sensitivity indices and a barplot with the ranked parameters. 
