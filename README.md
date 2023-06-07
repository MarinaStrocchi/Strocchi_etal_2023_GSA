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

These data 
