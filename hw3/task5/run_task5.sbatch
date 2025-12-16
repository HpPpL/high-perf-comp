#!/bin/bash
#SBATCH --job-name=hw3_task5
#SBATCH --output=task5_output_%j.txt
#SBATCH --error=task5_error_%j.txt
#SBATCH --time=00:30:00
#SBATCH --nodes=1
#SBATCH --ntasks=2
#SBATCH --gpus=2

# Load required modules
# Попробуем загрузить модули с обработкой ошибок
module load nvidia_sdk/nvhpc/23.5 2>&1 || echo "Warning: nvhpc module not found, trying alternative..."

# Проверяем доступность MPI из nvhpc
if command -v mpicc &> /dev/null; then
    echo "MPI found in PATH"
else
    # Пробуем найти MPI в nvhpc
    export PATH=/opt/software/nvidia/hpc_sdk/v23.5/Linux_x86_64/23.5/comm_libs/mpi/bin:$PATH
    export LD_LIBRARY_PATH=/opt/software/nvidia/hpc_sdk/v23.5/Linux_x86_64/23.5/comm_libs/mpi/lib:$LD_LIBRARY_PATH
fi

# Compile
make clean
make

# Run with 2 MPI processes (required)
echo "=== Running Task 5: Hybrid MPI+GPU ==="
if command -v srun &> /dev/null; then
    srun -n 2 --gpus=2 ./heat_equation 10000
else
    mpirun -np 2 ./heat_equation 10000
fi

