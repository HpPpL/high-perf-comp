#!/bin/bash
#SBATCH --job-name=task4_omp_profiling
#SBATCH --output=task4_%j.out
#SBATCH --error=task4_%j.err
#SBATCH --time=00:30:00
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=24
#SBATCH --mem=8G

# Load required modules
module load gcc/9.3.0
module load openmpi/4.1.0

# Set OpenMP environment variables
export OMP_NUM_THREADS=$SLURM_CPUS_PER_TASK
export OMP_PROC_BIND=close
export OMP_PLACES=cores

# Print job information
echo "Job ID: $SLURM_JOB_ID"
echo "Node: $SLURM_NODELIST"
echo "CPUs per task: $SLURM_CPUS_PER_TASK"
echo "OMP_NUM_THREADS: $OMP_NUM_THREADS"
echo "Start time: $(date)"
echo ""

# Compile the program
echo "Compiling task4..."
make clean
make task4

if [ $? -ne 0 ]; then
    echo "Compilation failed!"
    exit 1
fi

echo "Compilation successful!"
echo ""

# Run the program
echo "Running OpenMP matrix multiplication profiling..."
./task4

echo ""
echo "End time: $(date)"
echo "Job completed successfully!"

# Generate additional profiling if gprof is available
if command -v gprof &> /dev/null; then
    echo ""
    echo "Generating gprof profile report..."
    make task4_profile
    ./task4_profile
    gprof task4_profile gmon.out > gprof_report.txt
    echo "gprof report saved to gprof_report.txt"
fi
