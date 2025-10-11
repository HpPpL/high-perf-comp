#!/bin/bash
#SBATCH --job-name=Task1-seq-matrix-mult-makolodin
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=4
#SBATCH --time=00:10:00
#SBATCH --output=Task1-seq-matrix-mult-makolodin.out
#SBATCH --error=Task1-seq-matrix-mult-makolodin.err

module load cmake
module load gcc

mkdir -p build
cd build
cmake ..
make -j4

export OMP_NUM_THREADS=4
./task1
