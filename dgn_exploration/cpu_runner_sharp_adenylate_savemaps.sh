#!/bin/bash 

#PBS -l select=1:ncpus=1:mem=1gb

# imposta la coda di esecuzione
#PBS -q VARIAMOLS_cpuQ 
echo "loading modules"
#cd /home/marco.giulini/virtual_environment/env_conv3d/bin
#source activate
#module load cuda-8.0
#module load python-3.5.2

#echo "modules loaded"
#echo "running program"
#cd /home/marco.giulini/june_nn/convnet_models/gpu_models
export LD_LIBRARY_PATH=/home/marco.giulini/miniconda3/envs/test_pisa_trento/lib:$LD_LIBRARY_PATH
cd /home/marco.giulini/dgn_wl_exploration

source activate test_pisa_trento
#python3 dgn_WL.py
#python3 PredictTrentoDataset.py 
python3 dgn_wang_landau_adenylate_sharp_savemaps.py > "sharp_wang_landau_adenylate_savemaps.log"
echo "WL exploration finished"
