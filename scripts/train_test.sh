#!/bin/bash

## NOT PERMITTED  #SBATCH --partition=p_ccib_1              # Partition (job queue) 

#SBATCH --partition=main              # Partition (job queue) 
#SBATCH --no-requeue                 # Do not re-run job  if preempted
#SBATCH --job-name=train             # Assign an short name to your job
#SBATCH --nodes=1                    # Number of nodes you require
#SBATCH --ntasks=1                   # Total # of tasks across all nodes
#SBATCH --cpus-per-task=1            # Cores per task (>1 if multithread tasks)

# EITHER 1 or 2 works #SBATCH --gres=gpu:1                   # Number of GPUs

#SBATCH --gres=gpu:1                   # Number of GPUs
#SBATCH --mem=4096                   # Real memory (RAM) required (MB)
#SBATCH --constraint=pascal             # specifies to use the Pacal Node which has the P100 GPU
#SBATCH --time=01:00:00              # Total run time limit (HH:MM:SS)
#SBATCH --output=slurm.%N.%j.out     # STDOUT output file
#SBATCH --error=slurm.%N.%j.err      # STDERR output file (optional)
#SBATCH --export=ALL                 # Export you current env to the job env


cd ../../FragMap_CNN

python slurm_scripts/test.py

python train.py 

