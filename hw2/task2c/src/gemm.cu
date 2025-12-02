#include <cuda_runtime.h>
#include <cuda.h>
#include <iostream>
#include <iomanip>
#include <cmath>
#include <vector>
#include <chrono>
#include <fstream>

// CUDA kernel for matrix multiplication using shared memory (tiled)
__global__ void matrixMultiplyShared(double* A, double* B, double* C, int N) {
    // Shared memory for tiles
    __shared__ double tileA[16][16];
    __shared__ double tileB[16][16];
    
    // Thread indices
    int tx = threadIdx.x;
    int ty = threadIdx.y;
    int bx = blockIdx.x;
    int by = blockIdx.y;
    
    // Calculate row and column indices in output matrix
    int row = by * blockDim.y + ty;
    int col = bx * blockDim.x + tx;
    
    double sum = 0.0;
    
    // Iterate over tiles
    for (int tile = 0; tile < (N + blockDim.x - 1) / blockDim.x; ++tile) {
        // Load tile from A into shared memory
        int aRow = by * blockDim.y + ty;
        int aCol = tile * blockDim.x + tx;
        if (aRow < N && aCol < N) {
            tileA[ty][tx] = A[aCol * N + aRow];  // Column-major
        } else {
            tileA[ty][tx] = 0.0;
        }
        
        // Load tile from B into shared memory
        int bRow = tile * blockDim.y + ty;
        int bCol = bx * blockDim.x + tx;
        if (bRow < N && bCol < N) {
            tileB[ty][tx] = B[bCol * N + bRow];  // Column-major
        } else {
            tileB[ty][tx] = 0.0;
        }
        
        // Synchronize to ensure tiles are loaded
        __syncthreads();
        
        // Compute partial dot product
        for (int k = 0; k < blockDim.x; ++k) {
            sum += tileA[ty][k] * tileB[k][tx];
        }
        
        // Synchronize before loading next tile
        __syncthreads();
    }
    
    // Write result to global memory
    if (row < N && col < N) {
        C[col * N + row] = sum;  // Column-major
    }
}

// Verify results
bool verifyResults(double* C_gpu, double* C_cpu, int N, double tolerance = 1e-6) {
    double max_error = 0.0;
    for (int i = 0; i < N * N; ++i) {
        double error = std::abs(C_gpu[i] - C_cpu[i]);
        max_error = std::max(max_error, error);
        if (error > tolerance) {
            std::cout << "Mismatch at index " << i << ": GPU=" << C_gpu[i] 
                      << ", CPU=" << C_cpu[i] << ", error=" << error << std::endl;
            return false;
        }
    }
    std::cout << "Verification passed! Max error: " << max_error << std::endl;
    return true;
}

// CPU reference
void matrixMultiplyCPU(double* A, double* B, double* C, int N) {
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

// Initialize matrix
void initializeMatrix(double* matrix, int N) {
    for (int i = 0; i < N; ++i) {
        for (int j = 0; j < N; ++j) {
            matrix[j * N + i] = (double)(i + j) / N;
        }
    }
}

// Calculate GFLOPS
double calculateGFLOPS(int N, double time_ms) {
    double operations = 2.0 * N * N * N;
    return operations / (time_ms * 1e6);
}

int main(int argc, char* argv[]) {
    int N = 2048;
    if (argc > 1) {
        N = std::atoi(argv[1]);
    }
    
    std::cout << "=== CUDA Matrix Multiplication (Shared Memory - Tiled) ===" << std::endl;
    std::cout << "Matrix size: " << N << "x" << N << std::endl;
    std::cout << "Memory per matrix: " << (N * N * sizeof(double) / (1024.0 * 1024.0)) << " MB" << std::endl;
    
    // Get GPU properties
    cudaDeviceProp prop;
    cudaGetDeviceProperties(&prop, 0);
    std::cout << "GPU: " << prop.name << std::endl;
    std::cout << "Shared memory per block: " << (prop.sharedMemPerBlock / 1024.0) << " KB" << std::endl;
    
    // Allocate pinned memory
    size_t matrixSize = N * N * sizeof(double);
    double *h_A, *h_B, *h_C;
    cudaMallocHost((void**)&h_A, matrixSize);
    cudaMallocHost((void**)&h_B, matrixSize);
    cudaMallocHost((void**)&h_C, matrixSize);
    
    // Initialize matrices
    std::cout << "\nInitializing matrices..." << std::endl;
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
    
    // Configure kernel launch parameters
    // Use 16x16 thread blocks (tile size)
    int blockSize = 16;
    dim3 blockDim(blockSize, blockSize);
    dim3 gridDim((N + blockSize - 1) / blockSize, (N + blockSize - 1) / blockSize);
    
    std::cout << "Grid: (" << gridDim.x << ", " << gridDim.y << ")" << std::endl;
    std::cout << "Block: (" << blockDim.x << ", " << blockDim.y << ")" << std::endl;
    std::cout << "Shared memory per block: " << (2 * blockSize * blockSize * sizeof(double) / 1024.0) << " KB" << std::endl;
    
    // Warm up
    std::cout << "\nWarming up..." << std::endl;
    matrixMultiplyShared<<<gridDim, blockDim>>>(d_A, d_B, d_C, N);
    cudaDeviceSynchronize();
    
    // Create CUDA events for timing
    cudaEvent_t start, stop;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);
    
    // Benchmark
    const int numRuns = 10;
    float totalTime = 0.0f;
    
    std::cout << "Running " << numRuns << " iterations..." << std::endl;
    for (int run = 0; run < numRuns; ++run) {
        // CRITICAL: Synchronize before recording start event to ensure previous operations are complete
        cudaDeviceSynchronize();
        cudaEventRecord(start);
        matrixMultiplyShared<<<gridDim, blockDim>>>(d_A, d_B, d_C, N);
        
        // Check for kernel launch errors
        cudaError_t err = cudaGetLastError();
        if (err != cudaSuccess) {
            std::cerr << "CUDA kernel launch error: " << cudaGetErrorString(err) << std::endl;
            return 1;
        }
        
        cudaEventRecord(stop);
        cudaEventSynchronize(stop);
        
        float milliseconds = 0.0f;
        cudaEventElapsedTime(&milliseconds, start, stop);
        totalTime += milliseconds;
    }
    
    float avgTime = totalTime / numRuns;
    double gflops = calculateGFLOPS(N, avgTime);
    
    std::cout << "\n=== Results ===" << std::endl;
    std::cout << "Average time: " << std::fixed << std::setprecision(3) << avgTime << " ms" << std::endl;
    std::cout << "Performance: " << std::fixed << std::setprecision(2) << gflops << " GFLOPS" << std::endl;
    
    // Save results to CSV
    std::ofstream csvFile("task2c_results.csv", std::ios::app);
    bool fileExists = csvFile.tellp() > 0;
    if (!fileExists) {
        csvFile << "task,N,time_ms,gflops\n";
    }
    csvFile << "task2c," << N << "," << std::fixed << std::setprecision(3) 
            << avgTime << "," << std::setprecision(2) << gflops << "\n";
    csvFile.close();
    std::cout << "Results saved to task2c_results.csv" << std::endl;
    
    // Copy result back
    cudaMemcpy(h_C, d_C, matrixSize, cudaMemcpyDeviceToHost);
    
    // Verify results
    std::cout << "\nVerifying results..." << std::endl;
    double *C_cpu = new double[N * N];
    matrixMultiplyCPU(h_A, h_B, C_cpu, N);
    verifyResults(h_C, C_cpu, N);
    
    // Cleanup
    cudaFree(d_A);
    cudaFree(d_B);
    cudaFree(d_C);
    cudaFreeHost(h_A);
    cudaFreeHost(h_B);
    cudaFreeHost(h_C);
    delete[] C_cpu;
    cudaEventDestroy(start);
    cudaEventDestroy(stop);
    
    std::cout << "\nDone!" << std::endl;
    return 0;
}

