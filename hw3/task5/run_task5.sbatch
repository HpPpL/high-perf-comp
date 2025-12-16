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
echo "Loading NVIDIA HPC SDK module..."
if ! module load nvidia_sdk/nvhpc/23.5 2>/dev/null; then
    echo "Warning: nvidia_sdk/nvhpc/23.5 not found, trying alternative paths..."
    # Пробуем найти nvcc в стандартных путях
    if ! command -v nvcc &> /dev/null; then
        echo "Error: nvcc not found in PATH"
        exit 1
    fi
fi

# Проверяем доступность MPI из nvhpc
echo "Checking MPI availability..."
if command -v mpicc &> /dev/null; then
    MPI_CC=$(which mpicc)
    echo "MPI found in PATH: $MPI_CC"
    
    # Получаем пути к MPI через mpicc для передачи в Makefile
    MPI_INC=$(mpicc --showme:compile 2>/dev/null | grep -oP '(?<=-I)\S+' | head -1)
    MPI_LIB=$(mpicc --showme:link 2>/dev/null | grep -oP '(?<=-L)\S+' | head -1)
    
    if [ -n "$MPI_INC" ]; then
        export MPI_INCLUDE_PATH="$MPI_INC"
        echo "MPI include path: $MPI_INC"
    fi
    if [ -n "$MPI_LIB" ]; then
        export MPI_LIBRARY_PATH="$MPI_LIB"
        export LD_LIBRARY_PATH=$MPI_LIB:$LD_LIBRARY_PATH
        echo "MPI library path: $MPI_LIB"
    fi
else
    # Пробуем найти MPI в nvhpc
    NVHPC_PATH="/opt/software/nvidia/hpc_sdk/v23.5/Linux_x86_64/23.5/comm_libs/mpi"
    if [ -d "$NVHPC_PATH/bin" ]; then
        export PATH=$NVHPC_PATH/bin:$PATH
        export LD_LIBRARY_PATH=$NVHPC_PATH/lib:$LD_LIBRARY_PATH
        echo "MPI paths added from nvhpc"
    else
        # Пробуем альтернативный путь
        NVHPC_HPCX="/opt/software/nvidia/hpc_sdk/v23.5/Linux_x86_64/23.5/comm_libs/hpcx/hpcx-2.15/ompi"
        if [ -d "$NVHPC_HPCX/bin" ]; then
            export PATH=$NVHPC_HPCX/bin:$PATH
            export LD_LIBRARY_PATH=$NVHPC_HPCX/lib:$LD_LIBRARY_PATH
            echo "MPI paths added from hpcx"
        else
            echo "Warning: MPI not found, compilation may fail"
        fi
    fi
fi

# Compile
echo "Compiling Task 5..."
make clean
make

if [ $? -eq 0 ]; then
    echo "Compilation successful!"
    echo "=== Running Task 5: Hybrid MPI+GPU ==="
    
    # Проверяем наличие исполняемого файла
    if [ ! -f "./heat_equation" ]; then
        echo "Error: Executable heat_equation not found!"
        exit 1
    fi
    
    # Run with 2 MPI processes (required)
    if command -v srun &> /dev/null; then
        if srun -n 2 --gpus=2 ./heat_equation 10000; then
            echo "✓ Task 5 выполнена успешно"
        else
            echo "✗ Ошибка выполнения Task 5"
            exit 1
        fi
    else
        if mpirun -np 2 ./heat_equation 10000; then
            echo "✓ Task 5 выполнена успешно"
        else
            echo "✗ Ошибка выполнения Task 5"
            exit 1
        fi
    fi
else
    echo "Compilation failed!"
    exit 1
fi

