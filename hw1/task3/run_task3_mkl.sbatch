#!/bin/bash
#SBATCH --job-name=task3_mkl_comparison-makolodin
#SBATCH --output=task3_mkl-makolodin-%j.out
#SBATCH --error=task3_mkl-makolodin-%j.err
#SBATCH --time=00:30:00
#SBATCH --cpus-per-task=24
#SBATCH --nodes=1
#SBATCH --gpus=0
#SBATCH --ntasks=1

# Load required modules
module load intel/2020.4
module load gcc/9.3.0

# Try to load MKL module if available
module load intel-mkl 2>/dev/null || echo "intel-mkl module not found, trying alternative..."

# Alternative: try to set MKLROOT manually if module doesn't work
if [ -z "$MKLROOT" ]; then
    # Try common MKL paths
    if [ -d "/opt/intel/mkl" ]; then
        export MKLROOT="/opt/intel/mkl"
    elif [ -d "/usr/local/intel/mkl" ]; then
        export MKLROOT="/usr/local/intel/mkl"
    elif [ -d "$INTEL_HOME/mkl" ]; then
        export MKLROOT="$INTEL_HOME/mkl"
    fi
fi

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

# Check MKL environment
echo "=== Checking MKL environment ==="
echo "MKLROOT: $MKLROOT"
if [ -z "$MKLROOT" ]; then
    echo "Error: MKLROOT is not set. Make sure Intel module is loaded correctly."
    exit 1
fi

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
