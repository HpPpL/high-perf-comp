#!/bin/bash
#SBATCH --job-name=hw3_task1
#SBATCH --output=task1_output_%j.txt
#SBATCH --error=task1_error_%j.txt
#SBATCH --time=00:10:00
#SBATCH --nodes=1
#SBATCH --ntasks=1

# Load required modules
module load openmpi/4.1.5

# Compile
make clean
make

# Run
echo "=== Running Task 1: Algorithm Correctness Check ==="
mpirun -np 1 ./heat_equation

