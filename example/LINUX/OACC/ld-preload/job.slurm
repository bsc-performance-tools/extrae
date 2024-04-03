#!/usr/bin/env sh

#SBATCH --job-name=vecadd
#SBATCH --output=%x_%j.out
#SBATCH --error=%x_%j.err
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=80
#SBATCH --gres=gpu:2
#SBATCH --time=00:10:00

export OMP_NUM_THREADS=1

./trace.sh ./vecadd
