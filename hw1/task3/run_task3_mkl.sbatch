#!/bin/bash
#SBATCH --job-name=task3_mkl_comparison
#SBATCH --output=task3_mkl_%j.out
#SBATCH --error=task3_mkl_%j.err
#SBATCH --time=00:30:00
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=24
#SBATCH --mem=8G

# Load required modules
module load intel/2020.4
module load gcc/9.3.0

echo "=== Task 3: MKL Comparison ==="
echo "Job ID: $SLURM_JOB_ID"
echo "Node: $SLURM_NODELIST"
echo "CPUs: $SLURM_CPUS_PER_TASK"
echo "Memory: $SLURM_MEM_PER_NODE MB"
echo "Start time: $(date)"
echo

# Set environment variables
export OMP_NUM_THREADS=$SLURM_CPUS_PER_TASK
export MKL_NUM_THREADS=$SLURM_CPUS_PER_TASK

# Compile MKL version
echo "=== Compiling MKL version ==="
make clean
make task3_mkl

if [ $? -eq 0 ]; then
    echo "MKL compilation successful"
    echo
    
    # Run MKL benchmarks
    echo "=== Running MKL benchmarks ==="
    ./task3_mkl
    echo
    
    # Rename results file
    if [ -f "blas_comparison_results.csv" ]; then
        mv blas_comparison_results.csv mkl_results.csv
        echo "MKL results saved to mkl_results.csv"
    fi
else
    echo "MKL compilation failed!"
    exit 1
fi

echo
echo "=== Job completed ==="
echo "End time: $(date)"
echo "Results files:"
ls -la *.csv 2>/dev/null || echo "No CSV files found"
