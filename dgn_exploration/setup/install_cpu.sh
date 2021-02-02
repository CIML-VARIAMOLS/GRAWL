#!/bin/bash
#conda create -n GRAWL python=3.7
#conda activate GRAWL

conda_env=$( conda env list | grep "\*" | grep GRAWL | wc -l | bc )
echo "conda_env = ${conda_env}"
if [[ ${conda_env} == 0 ]]; then
	echo "GRAWL environment not successfully created or activated, aborting.."
	exit
elif [[ ${conda_env} == 1 ]]; then
	echo "GRAWL environment activated, installing packages..."
	if [[ "$OSTYPE" == "linux-gnu" ]]; then
	    pip install torch==1.4.0+cpu -f https://download.pytorch.org/whl/torch_stable.html
	elif [[ "$OSTYPE" == "darwin"* ]]; then
    	# Mac OSX
    	    pip install torch==1.4.0
	fi
	pip install -r requirements.txt
	pip install -r other_requirements_cpu.txt --find-links https://pytorch-geometric.com/whl/torch-1.4.0.html
fi
