#!/bin/bash
conda create -n GRAWL python=3.7
conda activate GRAWL

if [[ "$OSTYPE" == "linux-gnu" ]]; then
    pip install torch==1.4.0+cpu -f https://download.pytorch.org/whl/torch_stable.html
elif [[ "$OSTYPE" == "darwin"* ]]; then
    # Mac OSX
    pip install torch==1.4.0
fi

pip install -r requirements.txt
pip install -r other_requirements_cpu.txt --find-links https://pytorch-geometric.com/whl/torch-1.4.0.html
