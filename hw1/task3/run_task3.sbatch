#!/bin/bash
#SBATCH --job-name=task3_blas_comparison-makolodin
#SBATCH --output=task3-makolodin-%j.out
#SBATCH --error=task3-makolodin-%j.err
#SBATCH --time=00:30:00
#SBATCH --cpus-per-task=24
#SBATCH --nodes=1
#SBATCH --gpus=0
#SBATCH --ntasks=1

# Load required modules
module load OpenBlas/v0.3.18
module load gcc/9.3.0

echo "=== Task 3: BLAS Comparison ==="
echo "Job ID: $SLURM_JOB_ID"
echo "Node: $SLURM_NODELIST"
echo "CPUs: $SLURM_CPUS_PER_TASK"
echo "Memory: $SLURM_MEM_PER_NODE MB"
echo "Start time: $(date)"
echo

# Set environment variables
export OMP_NUM_THREADS=$SLURM_CPUS_PER_TASK
export OPENBLAS_NUM_THREADS=$SLURM_CPUS_PER_TASK

# Compile OpenBLAS version
echo "=== Compiling OpenBLAS version ==="
make clean
make task3_openblas

if [ $? -eq 0 ]; then
    echo "OpenBLAS compilation successful"
    echo
    
    # Run OpenBLAS benchmarks
    echo "=== Running OpenBLAS benchmarks ==="
    ./task3_openblas
    echo
    
    # Rename results file
    if [ -f "blas_comparison_results.csv" ]; then
        mv blas_comparison_results.csv openblas_results.csv
        echo "OpenBLAS results saved to openblas_results.csv"
    fi
else
    echo "OpenBLAS compilation failed!"
    exit 1
fi

echo
echo "=== Job completed ==="
echo "End time: $(date)"
echo "Results files:"
ls -la *.csv 2>/dev/null || echo "No CSV files found"
