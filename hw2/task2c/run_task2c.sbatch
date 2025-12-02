#!/bin/bash
#SBATCH --job-name=gemm_shared
#SBATCH --output=task2c_output_%j.txt
#SBATCH --error=task2c_error_%j.txt
#SBATCH --time=00:10:00
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gpus=1
#SBATCH --account=proj_1447

module load nvidia_sdk/nvhpc/23.5

make clean
make

echo "=== Running with N=1024 ==="
./gemm 1024

echo "=== Running with N=2048 ==="
./gemm 2048

