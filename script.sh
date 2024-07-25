#!/bin/bash -l
#SBATCH --time=00:30:00
#SBATCH --mem=10GB
#SBATCH --partition=gpu-debug
#SBATCH --gpus=1
#SBATCH --cpus-per-task=4
#SBATCH --output=.out/train.out
#SBATCH --error=.out/error.out

module load scicomp-python-env

python3 run.py -d cifar10 --arch resnet18 -t posterior-network -ld 6 -fl 6 --regr 1e-5 --lr 5e-4 -a fcr --cpus 4 --valfreq 10