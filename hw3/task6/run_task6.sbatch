#!/bin/bash
#SBATCH --job-name=hw3_task6
#SBATCH --output=task6_output_%j.txt
#SBATCH --error=task6_error_%j.txt
#SBATCH --time=00:30:00
#SBATCH --nodes=1
#SBATCH --ntasks=1

# Load required modules
# Пробуем разные варианты загрузки модулей Intel/MKL
module load intel/oneapi 2>&1 || \
module load intel/2021 2>&1 || \
module load mkl 2>&1 || \
echo "Warning: Intel/MKL module not found, trying direct paths..."

# Set MKL environment - пробуем разные пути
if [ -z "$MKLROOT" ]; then
    if [ -d "/opt/intel/oneapi/mkl/latest" ]; then
        export MKLROOT=/opt/intel/oneapi/mkl/latest
    elif [ -d "/opt/intel/mkl" ]; then
        export MKLROOT=/opt/intel/mkl
    elif [ -d "/usr/local/mkl" ]; then
        export MKLROOT=/usr/local/mkl
    fi
fi

if [ -n "$MKLROOT" ]; then
    export LD_LIBRARY_PATH=$MKLROOT/lib/intel64:$LD_LIBRARY_PATH
    echo "MKLROOT set to: $MKLROOT"
else
    echo "Warning: MKLROOT not found, compilation may fail"
fi

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

