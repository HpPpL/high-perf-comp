#!/bin/bash
#SBATCH --job-name=task1-dgemm-makolodin     
#SBATCH --error=task1_dgemm-makolodin-%j.err       
#SBATCH --output=task1_dgemm-makolodin-%j.log   
#SBATCH --time=00:05:00
#SBATCH --cpus-per-task=4
#SBATCH --nodes=1
#SBATCH --gpus=0

cd $SLURM_SUBMIT_DIR/src

echo "Compiling task1..."
g++ -O3 -fopenmp -o task1 task1.cpp

if [ $? -eq 0 ]; then
    echo "Compilation successful!"
    echo "Running task1..."
    time ./task1
else
    echo "Compilation failed!"
    exit 1
fi
