#!/bin/bash
#SBATCH --job-name=gemm_cublas
#SBATCH --output=task4_output_%j.txt
#SBATCH --error=task4_error_%j.txt
#SBATCH --time=00:10:00
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gpus=1
#SBATCH --account=proj_1447

module load nvidia_sdk/nvhpc/23.5

make clean
make

echo "=== Running cuBLAS with N=1024 ==="
./gemm 1024

echo "=== Running cuBLAS with N=2048 ==="
./gemm 2048

echo "=== Running cuBLAS with N=4096 ==="
./gemm 4096

