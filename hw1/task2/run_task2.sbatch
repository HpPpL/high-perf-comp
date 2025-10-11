#!/bin/bash
#SBATCH --job-name=task2-scalability-makolodin     
#SBATCH --error=task2_scalability-makolodin-%j.err       
#SBATCH --output=task2_scalability-makolodin-%j.log   
#SBATCH --time=00:30:00
#SBATCH --cpus-per-task=24
#SBATCH --nodes=1
#SBATCH --gpus=0

cd $SLURM_SUBMIT_DIR/src

echo "Compiling task2..."
g++ -O3 -fopenmp -o task2 task2.cpp

if [ $? -eq 0 ]; then
    echo "Compilation successful!"
    echo "Running scalability analysis..."
    time ./task2
    echo "Analysis completed!"
else
    echo "Compilation failed!"
    exit 1
fi
