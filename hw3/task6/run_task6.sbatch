#!/bin/bash
#SBATCH --job-name=hw3_task6
#SBATCH --output=task6_output_%j.txt
#SBATCH --error=task6_error_%j.txt
#SBATCH --time=00:30:00
#SBATCH --nodes=1
#SBATCH --ntasks=1

# Load required modules
# Пробуем разные варианты загрузки модулей Intel/MKL
echo "Loading Intel/MKL modules..."
MKL_LOADED=false

if module load intel/oneapi 2>/dev/null; then
    echo "Loaded: intel/oneapi"
    MKL_LOADED=true
elif module load intel/2021 2>/dev/null; then
    echo "Loaded: intel/2021"
    MKL_LOADED=true
elif module load mkl 2>/dev/null; then
    echo "Loaded: mkl"
    MKL_LOADED=true
else
    echo "Warning: Intel/MKL module not found, trying direct paths..."
fi

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
    
    # Проверяем наличие необходимых файлов
    if [ ! -d "$MKLROOT/include" ]; then
        echo "Warning: MKL include directory not found at $MKLROOT/include"
    fi
    if [ ! -d "$MKLROOT/lib/intel64" ]; then
        echo "Warning: MKL lib directory not found at $MKLROOT/lib/intel64"
    fi
else
    echo "Warning: MKLROOT not found, compilation may fail"
fi

# Compile
echo "Compiling Task 6..."
make clean
make

if [ $? -eq 0 ]; then
    echo "Compilation successful!"
    
    # Проверяем наличие исполняемого файла
    if [ ! -f "./heat_equation_2d" ]; then
        echo "Error: Executable heat_equation_2d not found!"
        exit 1
    fi
    
    # Initialize CSV file with header
    echo "N,execution_time" > task6_performance.csv
    
    # Run with different grid sizes
    echo "=== Running Task 6: 2D Heat Equation with MKL ==="
    success_count=0
    fail_count=0
    
    for N in 50 100 200; do
        echo "=== Running with N = $N ==="
        if ./heat_equation_2d $N; then
            echo "  ✓ Успешно выполнено для N=$N"
            ((success_count++))
        else
            echo "  ✗ Ошибка выполнения для N=$N"
            ((fail_count++))
        fi
    done
    
    echo ""
    echo "=== All tests completed ==="
    echo "  Успешно: $success_count"
    echo "  Ошибок: $fail_count"
    
    if [ $fail_count -gt 0 ]; then
        exit 1
    fi
else
    echo "Compilation failed!"
    exit 1
fi

