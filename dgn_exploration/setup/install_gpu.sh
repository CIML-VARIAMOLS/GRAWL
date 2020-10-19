#!/bin/bash
conda create -n GRAWL_gpu python=3.7
conda activate GRAWL_gpu

conda install -c pytorch=1.4.0+cu100 torchvision cudatoolkit=10.0 pytorch -f https://download.pytorch.org/whl/torch_stable.html

pip install -r requirements.txt
pip install -r other_requirements_gpu.txt --find-links https://pytorch-geometric.com/whl/torch-1.4.0.html
