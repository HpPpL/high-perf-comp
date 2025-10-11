#!/bin/bash
#SBATCH --job-name=Task1-seq-matrix-mult-makolodin
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=4
#SBATCH --time=00:10:00
#SBATCH --output=Task1-seq-matrix-mult-makolodin.out
#SBATCH --error=Task1-seq-matrix-mult-makolodin.err

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

# Диагностика модулей
echo "=== Available modules ==="
module avail 2>&1 | grep -E "(gcc|gnu|cmake)" | head -10

echo "=== CMake module info ==="
module spider cmake 2>&1 | head -20

echo "=== GCC module info ==="
module spider gcc 2>&1 | head -20

mkdir -p build
cd build

# Настраиваем CMake с явным указанием компилятора
cmake -DCMAKE_C_COMPILER=gcc -DCMAKE_CXX_COMPILER=g++ ..

make -j4

export OMP_NUM_THREADS=4
./task1
