#!/bin/bash -l
#SBATCH --time=24:00:00
#SBATCH --mem=40GB
#SBATCH --partition=gpu-p100-32g
#SBATCH --gpus=1
#SBATCH --cpus-per-task=4
#SBATCH --output=.out/train.out
#SBATCH --error=.out/error.out

module load scicomp-python-env

python3 run.py -d tin --arch resnet18 -t posterior-network -b 1024 -vb 1024 -ld 4 -fl 6 --regr 1e-5 --lr 1e-3 -a fcr --cpus 4 --valfreq 20