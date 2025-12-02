#!/bin/bash
#SBATCH --job-name=gemm_streams
#SBATCH --output=task2b_output_%j.txt
#SBATCH --error=task2b_error_%j.txt
#SBATCH --time=00:15:00
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gpus=1

module load nvidia_sdk/nvhpc/23.5

make clean
make

echo "=== Testing different number of streams (block size 16) ==="
./gemm 2048 2 16
./gemm 2048 4 16
./gemm 2048 8 16

echo "=== Testing different block sizes (4 streams) ==="
./gemm 2048 4 8
./gemm 2048 4 16
./gemm 2048 4 32

