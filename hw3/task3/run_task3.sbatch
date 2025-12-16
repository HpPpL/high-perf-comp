#!/bin/bash
#SBATCH --job-name=hw3_task3
#SBATCH --output=task3_output_%j.txt
#SBATCH --error=task3_error_%j.txt
#SBATCH --time=01:00:00
#SBATCH --nodes=1

# Load required modules
module load openmpi/4.1.5

# Compile
make clean
make

# Initialize CSV file with header
echo "num_procs,N,min_time,max_time,avg_time" > scalability_results_async.csv

# Test different numbers of processes: 1, 2, 4, 8, 16, 24
for num_procs in 1 2 4 8 16 24; do
    echo "=== Testing with $num_procs processes ==="
    
    # Test different grid sizes: 10000, 25000, 50000
    for N in 10000 25000 50000; do
        echo "  N = $N"
        mpirun -np $num_procs ./heat_equation $N
    done
done

echo "=== All tests completed ==="

