#!/bin/bash
#SBATCH --job-name=gemm_task1
#SBATCH --output=task1_output_%j.txt
#SBATCH --error=task1_error_%j.txt
#SBATCH --time=00:10:00
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gpus=1

# Load required modules
module load nvidia_sdk/nvhpc/23.5

# Compile
make clean
make

# Run with different matrix sizes
echo "=== Running with N=512 ==="
./gemm 512

echo "=== Running with N=1024 ==="
./gemm 1024

echo "=== Running with N=2048 ==="
./gemm 2048

