#include <cuda_runtime.h>
#include <cublas_v2.h>
#include <iostream>
#include <iomanip>
#include <cmath>
#include <vector>
#include <chrono>
#include <fstream>
#include <cstring>

#ifdef USE_OPENMP
#include <omp.h>
#endif

// ============================================================================
// CUDA KERNELS
// ============================================================================

// 1. Global memory GEMM (naive)
__global__ void gemm_global(double* A, double* B, double* C, int N) {
    int row = blockIdx.y * blockDim.y + threadIdx.y;
    int col = blockIdx.x * blockDim.x + threadIdx.x;
    
    if (row < N && col < N) {
        double sum = 0.0;
        // Column-major: A[k*N + i], B[j*N + k], C[j*N + i]
        for (int k = 0; k < N; ++k) {
            sum += A[k * N + row] * B[col * N + k];
        }
        C[col * N + row] = sum;
    }
}

// 2. Shared memory GEMM with tiling (TILE_SIZE x TILE_SIZE)
#define TILE_SIZE 16

__global__ void gemm_shared(double* A, double* B, double* C, int N) {
    // Shared memory for tiles
    __shared__ double tileA[TILE_SIZE][TILE_SIZE];
    __shared__ double tileB[TILE_SIZE][TILE_SIZE];
    
    int row = blockIdx.y * TILE_SIZE + threadIdx.y;
    int col = blockIdx.x * TILE_SIZE + threadIdx.x;
    
    double sum = 0.0;
    
    // Loop over tiles
    for (int tile = 0; tile < (N + TILE_SIZE - 1) / TILE_SIZE; ++tile) {
        // Load tile of A (column-major: A[k*N + i])
        int k = tile * TILE_SIZE + threadIdx.x;
        if (k < N && row < N) {
            tileA[threadIdx.y][threadIdx.x] = A[k * N + row];
        } else {
            tileA[threadIdx.y][threadIdx.x] = 0.0;
        }
        
        // Load tile of B (column-major: B[j*N + k])
        k = tile * TILE_SIZE + threadIdx.y;
        if (k < N && col < N) {
            tileB[threadIdx.y][threadIdx.x] = B[col * N + k];
        } else {
            tileB[threadIdx.y][threadIdx.x] = 0.0;
        }
        
        __syncthreads();
        
        // Compute partial dot product
        if (row < N && col < N) {
            for (int k = 0; k < TILE_SIZE; ++k) {
                sum += tileA[threadIdx.y][k] * tileB[k][threadIdx.x];
            }
        }
        
        __syncthreads();
    }
    
    // Write result
    if (row < N && col < N) {
        C[col * N + row] = sum;
    }
}

// 3. Shared memory GEMM with bank conflict optimization
// Use padding to avoid bank conflicts
#define TILE_SIZE_PADDED 17  // TILE_SIZE + 1 to avoid bank conflicts

__global__ void gemm_shared_optimized(double* A, double* B, double* C, int N) {
    // Shared memory with padding
    __shared__ double tileA[TILE_SIZE][TILE_SIZE_PADDED];
    __shared__ double tileB[TILE_SIZE][TILE_SIZE_PADDED];
    
    int row = blockIdx.y * TILE_SIZE + threadIdx.y;
    int col = blockIdx.x * TILE_SIZE + threadIdx.x;
    
    double sum = 0.0;
    
    // Loop over tiles
    for (int tile = 0; tile < (N + TILE_SIZE - 1) / TILE_SIZE; ++tile) {
        // Load tile of A
        int k = tile * TILE_SIZE + threadIdx.x;
        if (k < N && row < N) {
            tileA[threadIdx.y][threadIdx.x] = A[k * N + row];
        } else {
            tileA[threadIdx.y][threadIdx.x] = 0.0;
        }
        
        // Load tile of B
        k = tile * TILE_SIZE + threadIdx.y;
        if (k < N && col < N) {
            tileB[threadIdx.y][threadIdx.x] = B[col * N + k];
        } else {
            tileB[threadIdx.y][threadIdx.x] = 0.0;
        }
        
        __syncthreads();
        
        // Compute partial dot product
        if (row < N && col < N) {
            for (int k = 0; k < TILE_SIZE; ++k) {
                sum += tileA[threadIdx.y][k] * tileB[k][threadIdx.x];
            }
        }
        
        __syncthreads();
    }
    
    // Write result
    if (row < N && col < N) {
        C[col * N + row] = sum;
    }
}

// 4. Streams kernel - processes a tile of rows with shared memory tiling
__global__ void gemm_streams_tile(double* A, double* B, double* C, int N, 
                                   int tileRowStart, int tileRowEnd) {
    // Shared memory for tiles
    __shared__ double tileA[TILE_SIZE][TILE_SIZE];
    __shared__ double tileB[TILE_SIZE][TILE_SIZE];
    
    int row = blockIdx.y * TILE_SIZE + threadIdx.y + tileRowStart;
    int col = blockIdx.x * TILE_SIZE + threadIdx.x;
    
    double sum = 0.0;
    
    // Loop over tiles in K dimension
    int numTiles = (N + TILE_SIZE - 1) / TILE_SIZE;
    for (int tile = 0; tile < numTiles; ++tile) {
        // Load tile of A (column-major: A[k*N + i])
        int k = tile * TILE_SIZE + threadIdx.x;
        if (k < N && row < tileRowEnd) {
            tileA[threadIdx.y][threadIdx.x] = A[k * N + row];
        } else {
            tileA[threadIdx.y][threadIdx.x] = 0.0;
        }
        
        // Load tile of B (column-major: B[j*N + k])
        k = tile * TILE_SIZE + threadIdx.y;
        if (k < N && col < N) {
            tileB[threadIdx.y][threadIdx.x] = B[col * N + k];
        } else {
            tileB[threadIdx.y][threadIdx.x] = 0.0;
        }
        
        __syncthreads();
        
        // Compute partial dot product
        if (row < tileRowEnd && col < N) {
            for (int k = 0; k < TILE_SIZE; ++k) {
                sum += tileA[threadIdx.y][k] * tileB[k][threadIdx.x];
            }
        }
        
        __syncthreads();
    }
    
    // Write result
    if (row < tileRowEnd && col < N) {
        C[col * N + row] = sum;
    }
}

// ============================================================================
// CPU REFERENCE
// ============================================================================

void gemm_cpu(double* A, double* B, double* C, int N) {
    for (int i = 0; i < N; ++i) {
        for (int j = 0; j < N; ++j) {
            double sum = 0.0;
            for (int k = 0; k < N; ++k) {
                sum += A[k * N + i] * B[j * N + k];
            }
            C[j * N + i] = sum;
        }
    }
}

// ============================================================================
// UTILITY FUNCTIONS
// ============================================================================

void initializeMatrix(double* matrix, int N) {
    for (int i = 0; i < N; ++i) {
        for (int j = 0; j < N; ++j) {
            matrix[j * N + i] = (double)(i + j) / N;
        }
    }
}

bool verifyResults(double* C_gpu, double* C_cpu, int N, double tolerance = 1e-5) {
    double max_error = 0.0;
    int error_count = 0;
    for (int i = 0; i < N * N; ++i) {
        double error = std::abs(C_gpu[i] - C_cpu[i]);
        max_error = std::max(max_error, error);
        if (error > tolerance) {
            error_count++;
            if (error_count <= 5) {
                std::cout << "  Mismatch at index " << i << ": GPU=" << C_gpu[i] 
                          << ", CPU=" << C_cpu[i] << ", error=" << error << std::endl;
            }
        }
    }
    if (error_count > 0) {
        std::cout << "  ERROR: " << error_count << " mismatches found! Max error: " 
                  << max_error << std::endl;
        return false;
    }
    std::cout << "  Verification PASSED! Max error: " << max_error << std::endl;
    return true;
}

double calculateGFLOPS(int N, double time_ms) {
    double operations = 2.0 * N * N * N;
    return operations / (time_ms * 1e6);
}

// ============================================================================
// IMPLEMENTATION VARIANTS
// ============================================================================

#ifdef USE_GLOBAL
void run_global_memory(int N, int blockSize) {
    std::cout << "\n=== Global Memory GEMM ===" << std::endl;
    
    size_t matrixSize = N * N * sizeof(double);
    
    // Allocate pinned memory
    double *h_A, *h_B, *h_C, *h_C_cpu;
    cudaMallocHost((void**)&h_A, matrixSize);
    cudaMallocHost((void**)&h_B, matrixSize);
    cudaMallocHost((void**)&h_C, matrixSize);
    h_C_cpu = new double[N * N];
    
    initializeMatrix(h_A, N);
    initializeMatrix(h_B, N);
    
    // Allocate device memory
    double *d_A, *d_B, *d_C;
    cudaMalloc((void**)&d_A, matrixSize);
    cudaMalloc((void**)&d_B, matrixSize);
    cudaMalloc((void**)&d_C, matrixSize);
    
    // Copy to device
    cudaMemcpy(d_A, h_A, matrixSize, cudaMemcpyHostToDevice);
    cudaMemcpy(d_B, h_B, matrixSize, cudaMemcpyHostToDevice);
    
    // Launch kernel
    dim3 blockDim(blockSize, blockSize);
    dim3 gridDim((N + blockSize - 1) / blockSize, (N + blockSize - 1) / blockSize);
    
    // Warm up
    gemm_global<<<gridDim, blockDim>>>(d_A, d_B, d_C, N);
    cudaDeviceSynchronize();
    
    // Benchmark
    cudaEvent_t start, stop;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);
    
    const int numRuns = 10;
    float totalTime = 0.0f;
    
    for (int run = 0; run < numRuns; ++run) {
        cudaDeviceSynchronize();
        cudaEventRecord(start);
        gemm_global<<<gridDim, blockDim>>>(d_A, d_B, d_C, N);
        cudaEventRecord(stop);
        cudaEventSynchronize(stop);
        
        float ms = 0.0f;
        cudaEventElapsedTime(&ms, start, stop);
        totalTime += ms;
    }
    
    float avgTime = totalTime / numRuns;
    double gflops = calculateGFLOPS(N, avgTime);
    
    std::cout << "Time: " << std::fixed << std::setprecision(3) << avgTime << " ms" << std::endl;
    std::cout << "GFLOPS: " << std::setprecision(2) << gflops << std::endl;
    
    // Verify
    cudaMemcpy(h_C, d_C, matrixSize, cudaMemcpyDeviceToHost);
    gemm_cpu(h_A, h_B, h_C_cpu, N);
    verifyResults(h_C, h_C_cpu, N);
    
    // Save results
    std::ofstream csv("task2b_results.csv", std::ios::app);
    csv << "global," << N << "," << blockSize << "," << avgTime << "," << gflops << "\n";
    csv.close();
    
    // Cleanup
    cudaFree(d_A);
    cudaFree(d_B);
    cudaFree(d_C);
    cudaFreeHost(h_A);
    cudaFreeHost(h_B);
    cudaFreeHost(h_C);
    delete[] h_C_cpu;
    cudaEventDestroy(start);
    cudaEventDestroy(stop);
}
#endif

#ifdef USE_UNIFIED
void run_unified_memory(int N, int blockSize) {
    std::cout << "\n=== Unified Memory GEMM ===" << std::endl;
    
    size_t matrixSize = N * N * sizeof(double);
    
    // Allocate unified memory
    double *A, *B, *C, *C_cpu;
    cudaMallocManaged(&A, matrixSize);
    cudaMallocManaged(&B, matrixSize);
    cudaMallocManaged(&C, matrixSize);
    C_cpu = new double[N * N];
    
    initializeMatrix(A, N);
    initializeMatrix(B, N);
    
    // Launch kernel
    dim3 blockDim(blockSize, blockSize);
    dim3 gridDim((N + blockSize - 1) / blockSize, (N + blockSize - 1) / blockSize);
    
    // Warm up
    gemm_global<<<gridDim, blockDim>>>(A, B, C, N);
    cudaDeviceSynchronize();
    
    // Benchmark
    cudaEvent_t start, stop;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);
    
    const int numRuns = 10;
    float totalTime = 0.0f;
    
    for (int run = 0; run < numRuns; ++run) {
        cudaDeviceSynchronize();
        cudaEventRecord(start);
        gemm_global<<<gridDim, blockDim>>>(A, B, C, N);
        cudaEventRecord(stop);
        cudaEventSynchronize(stop);
        
        float ms = 0.0f;
        cudaEventElapsedTime(&ms, start, stop);
        totalTime += ms;
    }
    
    float avgTime = totalTime / numRuns;
    double gflops = calculateGFLOPS(N, avgTime);
    
    std::cout << "Time: " << std::fixed << std::setprecision(3) << avgTime << " ms" << std::endl;
    std::cout << "GFLOPS: " << std::setprecision(2) << gflops << std::endl;
    
    // Verify
    cudaDeviceSynchronize();
    gemm_cpu(A, B, C_cpu, N);
    verifyResults(C, C_cpu, N);
    
    // Save results
    std::ofstream csv("task2b_results.csv", std::ios::app);
    csv << "unified," << N << "," << blockSize << "," << avgTime << "," << gflops << "\n";
    csv.close();
    
    // Cleanup
    cudaFree(A);
    cudaFree(B);
    cudaFree(C);
    delete[] C_cpu;
    cudaEventDestroy(start);
    cudaEventDestroy(stop);
}
#endif

#ifdef USE_STREAMS
void run_streams(int N, int numStreams, int blockSize) {
    std::cout << "\n=== Streams GEMM (Pipeline: Copy → Compute → Copy) ===" << std::endl;
    std::cout << "Number of streams: " << numStreams << std::endl;
    
    size_t matrixSize = N * N * sizeof(double);
    int tileSize = (N + numStreams - 1) / numStreams;
    
    // Allocate pinned memory
    double *h_A, *h_B, *h_C, *h_C_cpu;
    cudaMallocHost((void**)&h_A, matrixSize);
    cudaMallocHost((void**)&h_B, matrixSize);
    cudaMallocHost((void**)&h_C, matrixSize);
    h_C_cpu = new double[N * N];
    
    initializeMatrix(h_A, N);
    initializeMatrix(h_B, N);
    
    // Allocate device memory
    double *d_A, *d_B, *d_C;
    cudaMalloc((void**)&d_A, matrixSize);
    cudaMalloc((void**)&d_B, matrixSize);
    cudaMalloc((void**)&d_C, matrixSize);
    
    // Copy B once (used by all streams)
    cudaMemcpy(d_B, h_B, matrixSize, cudaMemcpyHostToDevice);
    
    // Create streams
    std::vector<cudaStream_t> streams(numStreams);
    for (int i = 0; i < numStreams; ++i) {
        cudaStreamCreate(&streams[i]);
    }
    
    // Create events for timing
    cudaEvent_t start, stop;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);
    
    // Warm up
    for (int i = 0; i < numStreams; ++i) {
        int tileRowStart = i * tileSize;
        int tileRowEnd = std::min(tileRowStart + tileSize, N);
        size_t tileBytes = (tileRowEnd - tileRowStart) * N * sizeof(double);
        
        cudaMemcpyAsync(d_A + tileRowStart * N, h_A + tileRowStart * N, 
                       tileBytes, cudaMemcpyHostToDevice, streams[i]);
        
        dim3 blockDim(TILE_SIZE, TILE_SIZE);
        dim3 gridDim((N + TILE_SIZE - 1) / TILE_SIZE, 
                     (tileRowEnd - tileRowStart + TILE_SIZE - 1) / TILE_SIZE);
        gemm_streams_tile<<<gridDim, blockDim, 0, streams[i]>>>(
            d_A, d_B, d_C, N, tileRowStart, tileRowEnd);
    }
    cudaDeviceSynchronize();
    
    // Benchmark
    const int numRuns = 10;
    float totalTime = 0.0f;
    
    for (int run = 0; run < numRuns; ++run) {
        cudaDeviceSynchronize();
        cudaEventRecord(start);
        
        // Pipeline: For each stream, launch copy → compute → copy
        for (int i = 0; i < numStreams; ++i) {
            int tileRowStart = i * tileSize;
            int tileRowEnd = std::min(tileRowStart + tileSize, N);
            size_t tileBytes = (tileRowEnd - tileRowStart) * N * sizeof(double);
            
            // Phase 1: Async copy H2D
            cudaMemcpyAsync(d_A + tileRowStart * N, h_A + tileRowStart * N, 
                           tileBytes, cudaMemcpyHostToDevice, streams[i]);
            
            // Phase 2: Launch kernel in the same stream (waits for copy)
            dim3 blockDim(TILE_SIZE, TILE_SIZE);
            dim3 gridDim((N + TILE_SIZE - 1) / TILE_SIZE, 
                         (tileRowEnd - tileRowStart + TILE_SIZE - 1) / TILE_SIZE);
            gemm_streams_tile<<<gridDim, blockDim, 0, streams[i]>>>(
                d_A, d_B, d_C, N, tileRowStart, tileRowEnd);
            
            // Phase 3: Async copy D2H (waits for kernel)
            cudaMemcpyAsync(h_C + tileRowStart * N, d_C + tileRowStart * N, 
                           tileBytes, cudaMemcpyDeviceToHost, streams[i]);
        }
        
        // Wait for all streams
        for (int i = 0; i < numStreams; ++i) {
            cudaStreamSynchronize(streams[i]);
        }
        
        cudaEventRecord(stop);
        cudaEventSynchronize(stop);
        
        float ms = 0.0f;
        cudaEventElapsedTime(&ms, start, stop);
        totalTime += ms;
    }
    
    float avgTime = totalTime / numRuns;
    double gflops = calculateGFLOPS(N, avgTime);
    
    std::cout << "Time: " << std::fixed << std::setprecision(3) << avgTime << " ms" << std::endl;
    std::cout << "GFLOPS: " << std::setprecision(2) << gflops << std::endl;
    
    // Verify (results already copied)
    gemm_cpu(h_A, h_B, h_C_cpu, N);
    verifyResults(h_C, h_C_cpu, N);
    
    // Save results
    std::ofstream csv("task2b_results.csv", std::ios::app);
    csv << "streams," << N << "," << numStreams << "," << blockSize << "," 
        << avgTime << "," << gflops << "\n";
    csv.close();
    
    // Cleanup
    for (int i = 0; i < numStreams; ++i) {
        cudaStreamDestroy(streams[i]);
    }
    cudaFree(d_A);
    cudaFree(d_B);
    cudaFree(d_C);
    cudaFreeHost(h_A);
    cudaFreeHost(h_B);
    cudaFreeHost(h_C);
    delete[] h_C_cpu;
    cudaEventDestroy(start);
    cudaEventDestroy(stop);
}
#endif

#ifdef USE_SHARED
void run_shared_memory(int N) {
    std::cout << "\n=== Shared Memory GEMM (Tiled) ===" << std::endl;
    
    size_t matrixSize = N * N * sizeof(double);
    
    // Allocate pinned memory
    double *h_A, *h_B, *h_C, *h_C_cpu;
    cudaMallocHost((void**)&h_A, matrixSize);
    cudaMallocHost((void**)&h_B, matrixSize);
    cudaMallocHost((void**)&h_C, matrixSize);
    h_C_cpu = new double[N * N];
    
    initializeMatrix(h_A, N);
    initializeMatrix(h_B, N);
    
    // Allocate device memory
    double *d_A, *d_B, *d_C;
    cudaMalloc((void**)&d_A, matrixSize);
    cudaMalloc((void**)&d_B, matrixSize);
    cudaMalloc((void**)&d_C, matrixSize);
    
    // Copy to device
    cudaMemcpy(d_A, h_A, matrixSize, cudaMemcpyHostToDevice);
    cudaMemcpy(d_B, h_B, matrixSize, cudaMemcpyHostToDevice);
    
    // Launch kernel
    dim3 blockDim(TILE_SIZE, TILE_SIZE);
    dim3 gridDim((N + TILE_SIZE - 1) / TILE_SIZE, (N + TILE_SIZE - 1) / TILE_SIZE);
    
    // Warm up
    gemm_shared<<<gridDim, blockDim>>>(d_A, d_B, d_C, N);
    cudaDeviceSynchronize();
    
    // Benchmark
    cudaEvent_t start, stop;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);
    
    const int numRuns = 10;
    float totalTime = 0.0f;
    
    for (int run = 0; run < numRuns; ++run) {
        cudaDeviceSynchronize();
        cudaEventRecord(start);
        gemm_shared<<<gridDim, blockDim>>>(d_A, d_B, d_C, N);
        cudaEventRecord(stop);
        cudaEventSynchronize(stop);
        
        float ms = 0.0f;
        cudaEventElapsedTime(&ms, start, stop);
        totalTime += ms;
    }
    
    float avgTime = totalTime / numRuns;
    double gflops = calculateGFLOPS(N, avgTime);
    
    std::cout << "Time: " << std::fixed << std::setprecision(3) << avgTime << " ms" << std::endl;
    std::cout << "GFLOPS: " << std::setprecision(2) << gflops << std::endl;
    
    // Verify
    cudaMemcpy(h_C, d_C, matrixSize, cudaMemcpyDeviceToHost);
    gemm_cpu(h_A, h_B, h_C_cpu, N);
    verifyResults(h_C, h_C_cpu, N);
    
    // Save results
    std::ofstream csv("task2b_results.csv", std::ios::app);
    csv << "shared," << N << "," << TILE_SIZE << "," << avgTime << "," << gflops << "\n";
    csv.close();
    
    // Cleanup
    cudaFree(d_A);
    cudaFree(d_B);
    cudaFree(d_C);
    cudaFreeHost(h_A);
    cudaFreeHost(h_B);
    cudaFreeHost(h_C);
    delete[] h_C_cpu;
    cudaEventDestroy(start);
    cudaEventDestroy(stop);
}
#endif

#ifdef USE_SHARED_OPT
void run_shared_memory_optimized(int N) {
    std::cout << "\n=== Shared Memory GEMM (Optimized, Bank Conflict Free) ===" << std::endl;
    
    size_t matrixSize = N * N * sizeof(double);
    
    // Allocate pinned memory
    double *h_A, *h_B, *h_C, *h_C_cpu;
    cudaMallocHost((void**)&h_A, matrixSize);
    cudaMallocHost((void**)&h_B, matrixSize);
    cudaMallocHost((void**)&h_C, matrixSize);
    h_C_cpu = new double[N * N];
    
    initializeMatrix(h_A, N);
    initializeMatrix(h_B, N);
    
    // Allocate device memory
    double *d_A, *d_B, *d_C;
    cudaMalloc((void**)&d_A, matrixSize);
    cudaMalloc((void**)&d_B, matrixSize);
    cudaMalloc((void**)&d_C, matrixSize);
    
    // Copy to device
    cudaMemcpy(d_A, h_A, matrixSize, cudaMemcpyHostToDevice);
    cudaMemcpy(d_B, h_B, matrixSize, cudaMemcpyHostToDevice);
    
    // Launch kernel
    dim3 blockDim(TILE_SIZE, TILE_SIZE);
    dim3 gridDim((N + TILE_SIZE - 1) / TILE_SIZE, (N + TILE_SIZE - 1) / TILE_SIZE);
    
    // Warm up
    gemm_shared_optimized<<<gridDim, blockDim>>>(d_A, d_B, d_C, N);
    cudaDeviceSynchronize();
    
    // Benchmark
    cudaEvent_t start, stop;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);
    
    const int numRuns = 10;
    float totalTime = 0.0f;
    
    for (int run = 0; run < numRuns; ++run) {
        cudaDeviceSynchronize();
        cudaEventRecord(start);
        gemm_shared_optimized<<<gridDim, blockDim>>>(d_A, d_B, d_C, N);
        cudaEventRecord(stop);
        cudaEventSynchronize(stop);
        
        float ms = 0.0f;
        cudaEventElapsedTime(&ms, start, stop);
        totalTime += ms;
    }
    
    float avgTime = totalTime / numRuns;
    double gflops = calculateGFLOPS(N, avgTime);
    
    std::cout << "Time: " << std::fixed << std::setprecision(3) << avgTime << " ms" << std::endl;
    std::cout << "GFLOPS: " << std::setprecision(2) << gflops << std::endl;
    
    // Verify
    cudaMemcpy(h_C, d_C, matrixSize, cudaMemcpyDeviceToHost);
    gemm_cpu(h_A, h_B, h_C_cpu, N);
    verifyResults(h_C, h_C_cpu, N);
    
    // Save results
    std::ofstream csv("task2b_results.csv", std::ios::app);
    csv << "shared_opt," << N << "," << TILE_SIZE << "," << avgTime << "," << gflops << "\n";
    csv.close();
    
    // Cleanup
    cudaFree(d_A);
    cudaFree(d_B);
    cudaFree(d_C);
    cudaFreeHost(h_A);
    cudaFreeHost(h_B);
    cudaFreeHost(h_C);
    delete[] h_C_cpu;
    cudaEventDestroy(start);
    cudaEventDestroy(stop);
}
#endif

#ifdef USE_CUBLAS
void run_cublas(int N) {
    std::cout << "\n=== cuBLAS GEMM ===" << std::endl;
    
    size_t matrixSize = N * N * sizeof(double);
    
    // Allocate pinned memory
    double *h_A, *h_B, *h_C, *h_C_cpu;
    cudaMallocHost((void**)&h_A, matrixSize);
    cudaMallocHost((void**)&h_B, matrixSize);
    cudaMallocHost((void**)&h_C, matrixSize);
    h_C_cpu = new double[N * N];
    
    initializeMatrix(h_A, N);
    initializeMatrix(h_B, N);
    
    // Allocate device memory
    double *d_A, *d_B, *d_C;
    cudaMalloc((void**)&d_A, matrixSize);
    cudaMalloc((void**)&d_B, matrixSize);
    cudaMalloc((void**)&d_C, matrixSize);
    
    // Copy to device
    cudaMemcpy(d_A, h_A, matrixSize, cudaMemcpyHostToDevice);
    cudaMemcpy(d_B, h_B, matrixSize, cudaMemcpyHostToDevice);
    
    // Create cuBLAS handle
    cublasHandle_t handle;
    cublasCreate(&handle);
    
    const double alpha = 1.0, beta = 0.0;
    
    // Warm up
    // Note: Our matrices are stored in column-major: A[k*N + i] = A[i,k]
    // cuBLAS also uses column-major, so we can call directly
    // C = alpha * A * B + beta * C
    cublasDgemm(handle, CUBLAS_OP_N, CUBLAS_OP_N, N, N, N,
                &alpha, d_A, N, d_B, N, &beta, d_C, N);
    cudaDeviceSynchronize();
    
    // Benchmark
    cudaEvent_t start, stop;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);
    
    const int numRuns = 10;
    float totalTime = 0.0f;
    
    for (int run = 0; run < numRuns; ++run) {
        cudaDeviceSynchronize();
        cudaEventRecord(start);
        cublasDgemm(handle, CUBLAS_OP_N, CUBLAS_OP_N, N, N, N,
                    &alpha, d_A, N, d_B, N, &beta, d_C, N);
        cudaEventRecord(stop);
        cudaEventSynchronize(stop);
        
        float ms = 0.0f;
        cudaEventElapsedTime(&ms, start, stop);
        totalTime += ms;
    }
    
    float avgTime = totalTime / numRuns;
    double gflops = calculateGFLOPS(N, avgTime);
    
    std::cout << "Time: " << std::fixed << std::setprecision(3) << avgTime << " ms" << std::endl;
    std::cout << "GFLOPS: " << std::setprecision(2) << gflops << std::endl;
    
    // Verify
    cudaMemcpy(h_C, d_C, matrixSize, cudaMemcpyDeviceToHost);
    gemm_cpu(h_A, h_B, h_C_cpu, N);
    verifyResults(h_C, h_C_cpu, N);
    
    // Save results
    std::ofstream csv("task2b_results.csv", std::ios::app);
    csv << "cublas," << N << ",0," << avgTime << "," << gflops << "\n";
    csv.close();
    
    // Cleanup
    cublasDestroy(handle);
    cudaFree(d_A);
    cudaFree(d_B);
    cudaFree(d_C);
    cudaFreeHost(h_A);
    cudaFreeHost(h_B);
    cudaFreeHost(h_C);
    delete[] h_C_cpu;
    cudaEventDestroy(start);
    cudaEventDestroy(stop);
}
#endif

// ============================================================================
// MAIN
// ============================================================================

int main(int argc, char* argv[]) {
    int N = 2048;
    int numStreams = 4;
    int blockSize = 16;
    
    if (argc > 1) N = std::atoi(argv[1]);
    if (argc > 2) numStreams = std::atoi(argv[2]);
    if (argc > 3) blockSize = std::atoi(argv[3]);
    
    std::cout << "=== CUDA GEMM Benchmark Suite ===" << std::endl;
    std::cout << "Matrix size: " << N << "x" << N << std::endl;
    
    // Get GPU properties
    cudaDeviceProp prop;
    cudaGetDeviceProperties(&prop, 0);
    std::cout << "GPU: " << prop.name << std::endl;
    std::cout << "Compute Capability: " << prop.major << "." << prop.minor << std::endl;
    
    // Initialize CSV
    std::ofstream csv("task2b_results.csv", std::ios::app);
    if (csv.tellp() == 0) {
        csv << "variant,N,param1,param2,time_ms,gflops\n";
    }
    csv.close();
    
    // Run selected variants
#ifdef USE_GLOBAL
    run_global_memory(N, blockSize);
#endif

#ifdef USE_UNIFIED
    run_unified_memory(N, blockSize);
#endif

#ifdef USE_STREAMS
    run_streams(N, numStreams, blockSize);
#endif

#ifdef USE_SHARED
    run_shared_memory(N);
#endif

#ifdef USE_SHARED_OPT
    run_shared_memory_optimized(N);
#endif

#ifdef USE_CUBLAS
    run_cublas(N);
#endif

    std::cout << "\nDone!" << std::endl;
    return 0;
}
