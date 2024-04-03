#!/usr/bin/env sh

#SBATCH --job-name=pi
#SBATCH --output=%x_%j.out
#SBATCH --error=%x_%j.err
#SBATCH --ntasks=2
#SBATCH --cpus-per-task=80
#SBATCH --gres=gpu:2
#SBATCH --time=00:10:00

export OMP_NUM_THREADS=1

mpirun ./trace.sh ./pi
