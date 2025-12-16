#!/bin/bash
#SBATCH --job-name=hw3_task5
#SBATCH --output=task5_output_%j.txt
#SBATCH --error=task5_error_%j.txt
#SBATCH --time=00:30:00
#SBATCH --nodes=1
#SBATCH --ntasks=2
#SBATCH --gpus=2

# Load required modules
module load openmpi/4.1.5
module load nvidia_sdk/nvhpc/23.5

# Compile
make clean
make

# Run with 2 MPI processes (required)
echo "=== Running Task 5: Hybrid MPI+GPU ==="
mpirun -np 2 ./heat_equation 10000

