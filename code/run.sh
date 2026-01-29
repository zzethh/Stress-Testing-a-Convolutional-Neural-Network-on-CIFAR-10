#!/bin/bash
#SBATCH --job-name=cnn
#SBATCH --partition=mtech
#SBATCH --gres=gpu:1
#SBATCH --time=34:00:00
#SBATCH --output=logs.txt

module purge
module load python/3.10.pytorch

python3 train.py
python3 explain.py
