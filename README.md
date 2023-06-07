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
