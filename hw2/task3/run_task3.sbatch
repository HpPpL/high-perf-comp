#!/bin/bash
#SBATCH --job-name=gemm_banks
#SBATCH --output=task3_output_%j.txt
#SBATCH --error=task3_error_%j.txt
#SBATCH --time=00:10:00
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gpus=1
#SBATCH --account=proj_1447

module load nvidia_sdk/nvhpc/23.5

make clean
make

echo "=== Testing bank conflict optimization ==="
./gemm 2048

