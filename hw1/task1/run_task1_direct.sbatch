#!/bin/bash
#SBATCH --job-name=Task1-direct-makolodin
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=4
#SBATCH --time=00:10:00
#SBATCH --output=Task1-direct-makolodin.out
#SBATCH --error=Task1-direct-makolodin.err

# Загружаем модули
module load cmake

# Пробуем разные варианты загрузки компилятора
if module load gnu8 2>/dev/null; then
    echo "Loaded gnu8 module"
elif module load gcc/8.3.0 2>/dev/null; then
    echo "Loaded gcc/8.3.0 module"
elif module load gcc 2>/dev/null; then
    echo "Loaded gcc module"
else
    echo "Warning: Could not load GCC module, using system compiler"
fi

# Проверяем версии компиляторов
echo "GCC version:"
gcc --version
echo "G++ version:"
g++ --version

# Прямая компиляция без CMake
echo "Compiling directly with g++..."
g++ -O3 -fopenmp -o task1_direct task1.cpp

if [ -f "task1_direct" ]; then
    echo "Compilation successful!"
    export OMP_NUM_THREADS=4
    ./task1_direct
else
    echo "Compilation failed!"
    exit 1
fi
