#!/bin/bash
#SBATCH --job-name=hw3_task1
#SBATCH --output=task1_output_%j.txt
#SBATCH --error=task1_error_%j.txt
#SBATCH --time=00:10:00
#SBATCH --nodes=1
#SBATCH --ntasks=1

# Load required modules
module load openmpi/4.1.5 2>&1 || {
    echo "Warning: openmpi/4.1.5 not found, trying to find available MPI..."
    module avail openmpi 2>&1 | head -5
    exit 1
}

# Compile
make clean
make

# Run
echo "=== Running Task 1: Algorithm Correctness Check ==="
srun -n 1 ./heat_equation || mpirun -np 1 ./heat_equation

