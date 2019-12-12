#!/bin/bash 
#SBATCH --partition=main # Partition (job queue) 
#SBATCH --job-name=jupyter # Assign an short name to your job 
#SBATCH --nodes=1 # Number of nodes you require 
#SBATCH --ntasks=1 # Total # of tasks across all nodes 
#SBATCH --cpus-per-task=1 # Cores per task (>1 if multithread tasks) 

#SBATCH --requeue

#SBATCH --gres=gpu:1                   # Number of GPUs
#SBATCH --mem=4096                   # Real memory (RAM) required (MB)
#SBATCH --constraint=pascal             # specifies to use the Pacal Node which has the P10

#SBATCH --time=01:00:00 # Total run time limit (HH:MM:SS) 
#SBATCH --output=jupyter_outs/slurm.%N.%j.out # STDOUT output file 
#SBATCH --error=jupyter_errs/slurm.%N.%j.err # STDERR output file 
#(optional) export XDG_RUNTIME_DIR=$HOME/tmp ## needed for jupyter writting temporary files 
mkdir -p jupyter_outs
mkdir -p jupyter_errs

cd ../../FragMap_CNN


#DISPLAY=localhost:29.0

#echo $DISPLAY


srun jupyter notebook --no-browser --ip=0.0.0.0 --port=8889
