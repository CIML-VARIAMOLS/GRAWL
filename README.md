# README #

This is the official repository of the paper ["A deep graph network-enhanced sampling approach to efficiently explore the space of reduced representations of proteins"](https://www.frontiersin.org/articles/10.3389/fmolb.2021.637396/abstract), a joint collaboration between the Computational Intelligence and Machine Learning (CIML) group from the University of Pisa and the VARIAMOLS group from the University of Trento.

This repo allows to reproduce the main result of the paper (Wang Landau exploration of the space of CG mappings) and use the trained model for inference.

If you use this code, please remember to cite the paper:


    @article{ciml-variamols-2021,
      title={A deep graph network-enhanced sampling approach to efficiently explore the space of reduced representations of proteins},
      author={Errica, Federico and Giulini, Marco and Bacciu, Davide and Menichetti, Roberto and Micheli, Alessio and Potestio, Raffaello},
      journal={Frontiers in Molecular Biosciences},
      volume={8},
      pages={136},
      year={2021},
      publisher={Frontiers}
    }

We have also released the data used for this study.

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

## 1. setup a conda environment


1. create the GRAWL (deep GRAph network Wang Landau) conda environment:

	**conda create -n GRAWL python=3.7** (conda create -n GRAWL_gpu python=3.7)
	
2. activate the desired environment. This may vary as it depends on how your shell has been configured with conda (see [here](https://docs.conda.io/projects/conda/en/latest/user-guide/tasks/manage-environments.html#activating-an-environment))
	
	**conda activate GRAWL** (conda activate GRAWL_gpu)
	
3. set up the environment
	
	**./install_cpu.sh** (./install_gpu.sh)


## 2. running the Wang Landau exploration of the space

The python script *GRAWL.py* requires a parameter file in input. Two files "parameters.dat" are already present in the folder.
	
**python3 GRAWL.py parameters_4ake.dat**
	
**python3 GRAWL.py parameters_6d93.dat**
