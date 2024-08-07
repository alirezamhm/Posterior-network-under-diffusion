#!/bin/bash -l
#SBATCH --time=18:00:00
#SBATCH --mem=110GB
#SBATCH --partition=gpu-a100-80g
#SBATCH --gpus=1
#SBATCH --cpus-per-task=8
#SBATCH --output=.out/train.out
#SBATCH --error=.out/error.out

module load scicomp-python-env

python3 run.py -d tin --arch resnet18 -t posterior-network -b 4096 -vb 2000 -ld 16 -fl 10 --regr 1e-5 -wd 1e-6 --lr 5e-3 -a fcr --cpus 8 --valfreq 50