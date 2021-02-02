#!/bin/bash
#conda create -n GRAWL_gpu python=3.7
#conda activate GRAWL_gpu

conda_env=$( conda env list | grep "\*" | grep GRAWL_gpu | wc -l | bc )
echo "conda_env = ${conda_env}"

if [[ ${conda_env} == 0 ]]; then
        echo "GRAWL environment not successfully created or activated, aborting.."
        exit
elif [[ ${conda_env} == 1 ]]; then
        echo "GRAWL environment activated, installing packages..."
	conda install -c pytorch=1.4.0+cu100 torchvision cudatoolkit=10.0 pytorch -f https://download.pytorch.org/whl/torch_stable.html

	pip install -r requirements.txt
	pip install -r other_requirements_gpu.txt --find-links https://pytorch-geometric.com/whl/torch-1.4.0.html
fi
