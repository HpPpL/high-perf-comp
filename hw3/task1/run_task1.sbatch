#!/bin/bash
#SBATCH --job-name=hw3_task1
#SBATCH --output=task1_output_%j.txt
#SBATCH --error=task1_error_%j.txt
#SBATCH --time=00:10:00
#SBATCH --nodes=1
#SBATCH --ntasks=1

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
    # Run
    echo "=== Running Task 1: Algorithm Correctness Check ==="
    if srun -n 1 ./heat_equation 2>&1; then
        echo "✓ Task 1 выполнена успешно"
    else
        echo "Попытка запуска через mpirun..."
        mpirun -np 1 ./heat_equation 2>&1 || {
            echo "✗ Ошибка выполнения Task 1"
            exit 1
        }
    fi
else
    echo "Compilation failed!"
    exit 1
fi

