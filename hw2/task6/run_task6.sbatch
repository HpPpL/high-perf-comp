#!/bin/bash
#SBATCH --job-name=gemm_nsight
#SBATCH --output=task6_output_%j.txt
#SBATCH --error=task6_error_%j.txt
#SBATCH --time=00:15:00
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gpus=1

# Load required modules
module load nvidia_sdk/nvhpc/23.5

# Go to task2c directory (we profile the shared memory version)
cd ../task2c

# Compile
make clean
make

# Basic profiling
echo "=== Basic Profiling ==="
nsys profile --trace=cuda,nvtx --output=gemm_basic_profile ./gemm 2048

# Detailed GPU profiling
echo "=== Detailed GPU Profiling ==="
nsys profile --trace=cuda,nvtx --cuda-memory-usage=true \
    --output=gemm_detailed_profile ./gemm 2048

# Export statistics
echo "=== Exporting Statistics ==="
nsys stats --report gputrace gemm_detailed_profile.qdrep > gpu_stats.txt
nsys stats --report cudaapis gemm_detailed_profile.qdrep > cuda_apis.txt

echo "Profiling complete!"
echo "Results saved in task2c/ directory"
echo "Files created:"
echo "  - gemm_basic_profile.qdrep"
echo "  - gemm_detailed_profile.qdrep"
echo "  - gpu_stats.txt"
echo "  - cuda_apis.txt"

