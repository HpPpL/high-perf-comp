#!/bin/bash
#SBATCH --job-name=gemm_task2b
#SBATCH --output=task2b_output_%j.txt
#SBATCH --error=task2b_error_%j.txt
#SBATCH --time=00:30:00
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gpus=1

module load nvidia_sdk/nvhpc/23.5

make clean
make all

N=2048

echo "=========================================="
echo "Task 2b: Comprehensive GEMM Benchmark"
echo "=========================================="
echo "Matrix size: ${N}x${N}"
echo ""

# Remove old results
rm -f task2b_results.csv

echo "=== 1. Global Memory GEMM ==="
./gemm_global ${N} 4 16

echo ""
echo "=== 2. Unified Memory GEMM ==="
./gemm_unified ${N} 4 16

echo ""
echo "=== 3. Streams GEMM - Testing different number of streams ==="
echo "Testing with 2 streams..."
./gemm_streams ${N} 2 16
echo "Testing with 4 streams..."
./gemm_streams ${N} 4 16
echo "Testing with 8 streams..."
./gemm_streams ${N} 8 16
echo "Testing with 16 streams..."
./gemm_streams ${N} 16 16

echo ""
echo "=== 4. Streams GEMM - Testing different block sizes ==="
echo "Testing with block size 8..."
./gemm_streams ${N} 4 8
echo "Testing with block size 16..."
./gemm_streams ${N} 4 16
echo "Testing with block size 32..."
./gemm_streams ${N} 4 32

echo ""
echo "=== 5. Shared Memory GEMM (Tiled) ==="
./gemm_shared ${N} 4 16

echo ""
echo "=== 6. Shared Memory GEMM (Optimized, Bank Conflict Free) ==="
./gemm_shared_opt ${N} 4 16

echo ""
echo "=== 7. cuBLAS GEMM (Reference) ==="
./gemm_cublas ${N} 4 16

echo ""
echo "=========================================="
echo "Benchmark completed!"
echo "Results saved to task2b_results.csv"
echo "=========================================="
