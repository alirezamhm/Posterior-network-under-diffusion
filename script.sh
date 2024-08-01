#!/bin/bash -l
#SBATCH --time=18:00:00
#SBATCH --mem=100GB
#SBATCH --partition=gpu-v100-32g
#SBATCH --gpus=1
#SBATCH --cpus-per-task=10
#SBATCH --output=.out/train.out
#SBATCH --error=.out/error.out

module load scicomp-python-env
 
python3 run.py -d tin --arch resnet18 -t posterior-network -b 1024 -vb 1000 -ld 4 -fl 6 --regr 1e-5 --lr 1e-3 -a fcr --cpus 10 --valfreq 50