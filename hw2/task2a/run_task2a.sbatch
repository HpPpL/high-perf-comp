#!/bin/bash
#SBATCH --job-name=gemm_unified
#SBATCH --output=task2a_output_%j.txt
#SBATCH --error=task2a_error_%j.txt
#SBATCH --time=00:10:00
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gpus=1

module load nvidia_sdk/nvhpc/23.5

make clean
make

echo "=== Running with N=1024 ==="
./gemm 1024

echo "=== Running with N=2048 ==="
./gemm 2048

