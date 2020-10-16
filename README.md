# README #

This README would normally document whatever steps are necessary to get your application up and running.

### What is this repository for? ###

* produce a dataset of all mappings tried with the corresponding mapping entropies, embedded in the form of a graph
* include all the analysis
* a bit of literature on ML for graphs

## folders

* dataset
	- raw_data
	- cleaned_data
* literature
* dgn_exploration

This folder contains the trained neural network and the python script to perform Wang Landau exploration in the space of available mappings.

In order to use them:

1. setup a conda environment HOW

2. run the script: the two files "parameters.dat" are already present in the folder and should be the first and unique argument given to the script:
	
	**python3 GRAWL.py parameters.dat**
