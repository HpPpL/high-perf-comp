#!/bin/bash
#SBATCH --job-name=gemm_omp
#SBATCH --output=task5_output_%j.txt
#SBATCH --error=task5_error_%j.txt
#SBATCH --time=00:10:00
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gpus=1
#SBATCH --account=proj_1447

module load nvidia_sdk/nvhpc/23.5

make clean
make

echo "=== Running OpenMP GPU with N=1024 ==="
./gemm 1024

echo "=== Running OpenMP GPU with N=2048 ==="
./gemm 2048

