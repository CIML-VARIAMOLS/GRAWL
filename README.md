# README #

This README would normally document whatever steps are necessary to get your application up and running.

### What is this repository for? ###

* dataset used in the article "A deep graph network-enhanced sampling approach to efficiently explore the space of reduced representations of proteins"
* the setup scripts for the Wang Landau exploration of the space of CG mappings 

## folders

* dataset
* dgn_exploration

# dataset

A folder that contains five files:

- 2_graphs_dataset.dat: it contains the structure of the two input graphs for 6d93 and 4ake
- 4ake_smaps_def_scaled.txt : values of smap for 4ake
- 6d93_smaps_def_scaled.txt : values of smap for 6d93
- 4ake_mappings_def.txt: mappings for 4ake
- 6d93_mappings_def.txt: mappings for 6d93

# dgn_exploration

This folder contains the trained neural network and the python script to perform Wang Landau exploration in the space of available mappings.

In order to use them:

1. setup a conda environment. There are two scripts (install_cpu.sh and install_gpu.sh) that set up a conda environment depending on the device used.

	**Usage**: running *./install_cpu.sh* will create GRAWL, a cpu-based environment, while running *./install_gpu.sh* will create the GRAWL_gpu environment, with exactly the same packages. GRAWL_gpu is created (and tested) with compatibility with cuda 10.0, but the version of cuda can be easily modified inside the script. 

2. run the script: the two files "parameters.dat" are already present in the folder. The name of the parameter file should be the first and unique argument given to the script:
	
	**python3 GRAWL.py parameters_4ake.dat**
	
	**python3 GRAWL.py parameters_6d93.dat**
