#!/bin/bash
#SBATCH --job-name=hw3_task6
#SBATCH --output=task6_output_%j.txt
#SBATCH --error=task6_error_%j.txt
#SBATCH --time=00:30:00
#SBATCH --nodes=1
#SBATCH --ntasks=1

# Load required modules
module load intel/oneapi

# Set MKL environment
export MKLROOT=/opt/intel/oneapi/mkl/latest
export LD_LIBRARY_PATH=$MKLROOT/lib/intel64:$LD_LIBRARY_PATH

# Compile
make clean
make

# Initialize CSV file with header
echo "N,execution_time" > task6_performance.csv

# Run with different grid sizes
for N in 50 100 200; do
    echo "=== Running with N = $N ==="
    ./heat_equation_2d $N
done

echo "=== All tests completed ==="

