#include <cuda_runtime.h>
#include <cuda.h>
#include <iostream>
#include <iomanip>
#include <cmath>
#include <vector>
#include <chrono>
#include <fstream>

// CUDA kernel with potential bank conflicts (no padding)
__global__ void matrixMultiplySharedNoPadding(double* A, double* B, double* C, int N) {
    __shared__ double tileA[16][16];
    __shared__ double tileB[16][16];
    
    int tx = threadIdx.x;
    int ty = threadIdx.y;
    int bx = blockIdx.x;
    int by = blockIdx.y;
    
    int row = by * blockDim.y + ty;
    int col = bx * blockDim.x + tx;
    
    double sum = 0.0;
    
    for (int tile = 0; tile < (N + blockDim.x - 1) / blockDim.x; ++tile) {
        int aRow = by * blockDim.y + ty;
        int aCol = tile * blockDim.x + tx;
        if (aRow < N && aCol < N) {
            tileA[ty][tx] = A[aCol * N + aRow];
        } else {
            tileA[ty][tx] = 0.0;
        }
        
        int bRow = tile * blockDim.y + ty;
        int bCol = bx * blockDim.x + tx;
        if (bRow < N && bCol < N) {
            tileB[ty][tx] = B[bCol * N + bRow];
        } else {
            tileB[ty][tx] = 0.0;
        }
        
        __syncthreads();
        
        for (int k = 0; k < blockDim.x; ++k) {
            sum += tileA[ty][k] * tileB[k][tx];
        }
        
        __syncthreads();
    }
    
    if (row < N && col < N) {
        C[col * N + row] = sum;
    }
}

// CUDA kernel with bank conflict avoidance (padding)
__global__ void matrixMultiplySharedPadded(double* A, double* B, double* C, int N) {
    // Add padding to avoid bank conflicts (17 elements per row instead of 16)
    __shared__ double tileA[16][17];
    __shared__ double tileB[16][17];
    
    int tx = threadIdx.x;
    int ty = threadIdx.y;
    int bx = blockIdx.x;
    int by = blockIdx.y;
    
    int row = by * blockDim.y + ty;
    int col = bx * blockDim.x + tx;
    
    double sum = 0.0;
    
    for (int tile = 0; tile < (N + blockDim.x - 1) / blockDim.x; ++tile) {
        int aRow = by * blockDim.y + ty;
        int aCol = tile * blockDim.x + tx;
        if (aRow < N && aCol < N) {
            tileA[ty][tx] = A[aCol * N + aRow];
        } else {
            tileA[ty][tx] = 0.0;
        }
        
        int bRow = tile * blockDim.y + ty;
        int bCol = bx * blockDim.x + tx;
        if (bRow < N && bCol < N) {
            tileB[ty][tx] = B[bCol * N + bRow];
        } else {
            tileB[ty][tx] = 0.0;
        }
        
        __syncthreads();
        
        for (int k = 0; k < blockDim.x; ++k) {
            sum += tileA[ty][k] * tileB[k][tx];
        }
        
        __syncthreads();
    }
    
    if (row < N && col < N) {
        C[col * N + row] = sum;
    }
}

// Calculate GFLOPS
double calculateGFLOPS(int N, double time_ms) {
    double operations = 2.0 * N * N * N;
    return operations / (time_ms * 1e6);
}

// Benchmark a kernel
float benchmarkKernel(void (*kernel)(double*, double*, double*, int),
                      double* d_A, double* d_B, double* d_C, int N,
                      dim3 gridDim, dim3 blockDim, int numRuns) {
    cudaEvent_t start, stop;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);
    
    // Warm up
    kernel<<<gridDim, blockDim>>>(d_A, d_B, d_C, N);
    cudaDeviceSynchronize();
    
    float totalTime = 0.0f;
    for (int run = 0; run < numRuns; ++run) {
        // CRITICAL: Synchronize before recording start event to ensure previous operations are complete
        cudaDeviceSynchronize();
        cudaEventRecord(start);
        kernel<<<gridDim, blockDim>>>(d_A, d_B, d_C, N);
        
        // Check for kernel launch errors
        cudaError_t err = cudaGetLastError();
        if (err != cudaSuccess) {
            std::cerr << "CUDA kernel launch error: " << cudaGetErrorString(err) << std::endl;
            return -1.0f;
        }
        
        cudaEventRecord(stop);
        cudaEventSynchronize(stop);
        
        float milliseconds = 0.0f;
        cudaEventElapsedTime(&milliseconds, start, stop);
        totalTime += milliseconds;
    }
    
    cudaEventDestroy(start);
    cudaEventDestroy(stop);
    
    return totalTime / numRuns;
}

int main(int argc, char* argv[]) {
    int N = 2048;
    if (argc > 1) {
        N = std::atoi(argv[1]);
    }
    
    std::cout << "=== Shared Memory Bank Conflict Optimization ===" << std::endl;
    std::cout << "Matrix size: " << N << "x" << N << std::endl;
    
    // Get GPU properties
    cudaDeviceProp prop;
    cudaGetDeviceProperties(&prop, 0);
    std::cout << "GPU: " << prop.name << std::endl;
    std::cout << "Shared memory banks: 32" << std::endl;
    std::cout << "Bank width: 4 bytes (32-bit)" << std::endl;
    
    // Allocate memory
    size_t matrixSize = N * N * sizeof(double);
    double *h_A, *h_B;
    cudaMallocHost((void**)&h_A, matrixSize);
    cudaMallocHost((void**)&h_B, matrixSize);
    
    // Initialize matrices
    for (int i = 0; i < N; ++i) {
        for (int j = 0; j < N; ++j) {
            h_A[j * N + i] = (double)(i + j) / N;
            h_B[j * N + i] = (double)(i * j) / (N * N);
        }
    }
    
    double *d_A, *d_B, *d_C;
    cudaMalloc((void**)&d_A, matrixSize);
    cudaMalloc((void**)&d_B, matrixSize);
    cudaMalloc((void**)&d_C, matrixSize);
    
    cudaMemcpy(d_A, h_A, matrixSize, cudaMemcpyHostToDevice);
    cudaMemcpy(d_B, h_B, matrixSize, cudaMemcpyHostToDevice);
    
    // Configure kernel launch
    int blockSize = 16;
    dim3 blockDim(blockSize, blockSize);
    dim3 gridDim((N + blockSize - 1) / blockSize, (N + blockSize - 1) / blockSize);
    
    const int numRuns = 10;
    
    std::cout << "\n=== Benchmarking ===" << std::endl;
    std::cout << "Grid: (" << gridDim.x << ", " << gridDim.y << ")" << std::endl;
    std::cout << "Block: (" << blockDim.x << ", " << blockDim.y << ")" << std::endl;
    
    // Test version without padding (potential bank conflicts)
    std::cout << "\n1. Testing without padding (potential bank conflicts)..." << std::endl;
    float timeNoPadding = benchmarkKernel(matrixMultiplySharedNoPadding, 
                                          d_A, d_B, d_C, N, gridDim, blockDim, numRuns);
    if (timeNoPadding < 0.0f) {
        std::cerr << "Error during benchmark (no padding)" << std::endl;
        return 1;
    }
    double gflopsNoPadding = calculateGFLOPS(N, timeNoPadding);
    
    // Test version with padding (bank conflict avoidance)
    std::cout << "2. Testing with padding (bank conflict avoidance)..." << std::endl;
    float timePadded = benchmarkKernel(matrixMultiplySharedPadded, 
                                       d_A, d_B, d_C, N, gridDim, blockDim, numRuns);
    if (timePadded < 0.0f) {
        std::cerr << "Error during benchmark (with padding)" << std::endl;
        return 1;
    }
    double gflopsPadded = calculateGFLOPS(N, timePadded);
    
    // Results
    std::cout << "\n=== Results ===" << std::endl;
    std::cout << std::fixed << std::setprecision(3);
    std::cout << "Without padding: " << timeNoPadding << " ms, " 
              << std::setprecision(2) << gflopsNoPadding << " GFLOPS" << std::endl;
    std::cout << std::setprecision(3);
    std::cout << "With padding:   " << timePadded << " ms, " 
              << std::setprecision(2) << gflopsPadded << " GFLOPS" << std::endl;
    
    float speedup = timeNoPadding / timePadded;
    std::cout << "\nSpeedup: " << std::setprecision(3) << speedup << "x" << std::endl;
    
    if (speedup > 1.0) {
        std::cout << "Padding improves performance by " << ((speedup - 1.0) * 100.0) << "%" << std::endl;
    } else {
        std::cout << "Padding does not improve performance (overhead may be higher than bank conflicts)" << std::endl;
    }
    
    // Save results to CSV
    std::ofstream csvFile("task3_results.csv", std::ios::app);
    bool fileExists = csvFile.tellp() > 0;
    if (!fileExists) {
        csvFile << "task,N,version,time_ms,gflops\n";
    }
    csvFile << "task3," << N << ",no_padding," << std::fixed << std::setprecision(3) 
            << timeNoPadding << "," << std::setprecision(2) << gflopsNoPadding << "\n";
    csvFile << "task3," << N << ",with_padding," << std::fixed << std::setprecision(3) 
            << timePadded << "," << std::setprecision(2) << gflopsPadded << "\n";
    csvFile.close();
    std::cout << "Results saved to task3_results.csv" << std::endl;
    
    // Cleanup
    cudaFree(d_A);
    cudaFree(d_B);
    cudaFree(d_C);
    cudaFreeHost(h_A);
    cudaFreeHost(h_B);
    
    std::cout << "\nDone!" << std::endl;
    return 0;
}

