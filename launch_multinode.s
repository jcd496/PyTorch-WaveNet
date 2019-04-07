#!/bin/bash

#SBATCH --job-name=aemulus
#SBATCH --nodes=1
#SBATCH --output=out/sicn_e9000_NPL1024.out
#SBATCH --error=out/redo_er
#SBATCH --cpus-per-task=1
#SBATCH --time=15:00:00
#SBATCH --mem=10gb
###SBATCH --gres=gpu:1

module purge
module load openmpi/intel/2.0.3
module load cuda/9.2.88
module load anaconda3/5.3.0

. /share/apps/anaconda3/5.3.0/etc/profile.d/conda.sh

conda activate dis_pytorch_env





