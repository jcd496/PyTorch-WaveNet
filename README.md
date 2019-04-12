# ParallelWaveNet

Parallel implementation of DeepMind's WaveNet utilizing PyTorch

by: Jacqueline Abalo and John Donaghy

Dependencies:

pytorch, torchvision, librosa

enter interactive session with
srun --mem=5gb --time=4:00:00 --gres=gpu:1 --pty -c 4 /bin/bash

conda activate dis_pytorch_env

conda install pytorch torchvision cudatoolkit=10.0 -c pytorch

to launch training from log-0 node (don't launch while in interactive session)
sbatch launch_multinode.s

Data Processed with preprocessing script from ksw0306 FloWaveNet.
Data Location on prince: /scratch/jcd496/LJdata/processed
