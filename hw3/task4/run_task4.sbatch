#!/bin/bash
#SBATCH --job-name=hw3_task4
#SBATCH --output=task4_output_%j.txt
#SBATCH --error=task4_error_%j.txt
#SBATCH --time=01:00:00
#SBATCH --nodes=1
#SBATCH --ntasks=24
#SBATCH --cpus-per-task=1

# Load required modules
# Пробуем загрузить разные версии MPI
if ! module load openmpi/4.1.5 2>/dev/null; then
    echo "Warning: openmpi/4.1.5 not found, trying openmpi3/3.1.4..."
    if ! module load openmpi3/3.1.4 2>/dev/null; then
        echo "Warning: openmpi3/3.1.4 not found, trying to find available MPI..."
        module avail openmpi 2>&1 | head -10
        # Пробуем загрузить первый доступный openmpi3
        module load openmpi3 2>/dev/null || {
            echo "Error: No MPI module found!"
            exit 1
        }
    fi
fi

# Compile
make clean
make

if [ $? -eq 0 ]; then
    echo "Compilation successful!"
    
    # Initialize CSV file with header
    echo "num_procs,N,min_time,max_time,avg_time" > scalability_results_collective.csv
    
    # Test different numbers of processes: 1, 2, 4, 8, 16, 24
    for num_procs in 1 2 4 8 16 24; do
        echo "=== Testing with $num_procs processes ==="
        
        # Test different grid sizes: 10000, 25000, 50000
        for N in 10000 25000 50000; do
            echo "  N = $N"
            if srun -n $num_procs ./heat_equation $N; then
                echo "  ✓ Успешно выполнено для N=$N с $num_procs процессами"
            else
                echo "  ✗ Ошибка выполнения для N=$N с $num_procs процессами"
            fi
        done
    done
else
    echo "Compilation failed!"
    exit 1
fi

echo "=== All tests completed ==="

