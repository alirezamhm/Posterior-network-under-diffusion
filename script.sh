#!/bin/bash -l
#SBATCH --time=12:00:00
#SBATCH --mem=32GB
#SBATCH --partition=gpu-p100-16g
#SBATCH --gpus=1
#SBATCH --cpus-per-task=6
#SBATCH --output=.out/train.out
#SBATCH --error=.out/error.out

module load scicomp-python-env

python3 run.py -d tin --arch resnet18 -t baseline --lr 1e-3 -a fcr --cpus 6 --valfreq 20