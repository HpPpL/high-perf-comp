#include <cuda_runtime.h>
#include <cuda.h>
#include <iostream>
#include <iomanip>
#include <cmath>
#include <vector>
#include <chrono>
#include <fstream>

// CUDA kernel for matrix multiplication using global memory
__global__ void matrixMultiplyGlobal(double* A, double* B, double* C, int N) {
    // Calculate row and column indices
    int row = blockIdx.y * blockDim.y + threadIdx.y;
    int col = blockIdx.x * blockDim.x + threadIdx.x;
    
    if (row < N && col < N) {
        double sum = 0.0;
        // Column-major order: A[k*N + i], B[j*N + k], C[j*N + i]
        for (int k = 0; k < N; ++k) {
            sum += A[k * N + row] * B[col * N + k];
        }
        C[col * N + row] = sum;
    }
}

// Verify results by comparing with CPU computation
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

// CPU reference implementation
void matrixMultiplyCPU(double* A, double* B, double* C, int N) {
    for (int i = 0; i < N; ++i) {
        for (int j = 0; j < N; ++j) {
            double sum = 0.0;
            for (int k = 0; k < N; ++k) {
                // Column-major: A[k*N + i], B[j*N + k]
                sum += A[k * N + i] * B[j * N + k];
            }
            C[j * N + i] = sum;
        }
    }
}

// Initialize matrix with test data
void initializeMatrix(double* matrix, int N) {
    for (int i = 0; i < N; ++i) {
        for (int j = 0; j < N; ++j) {
            matrix[j * N + i] = (double)(i + j) / N;
        }
    }
}

// Calculate GFLOPS
double calculateGFLOPS(int N, double time_ms) {
    // DGEMM: 2*N^3 floating point operations
    double operations = 2.0 * N * N * N;
    return operations / (time_ms * 1e6); // Convert milliseconds to seconds
}

int main(int argc, char* argv[]) {
    // Default matrix size
    int N = 1024;
    if (argc > 1) {
        N = std::atoi(argv[1]);
    }
    
    std::cout << "=== CUDA Matrix Multiplication (Global Memory) ===" << std::endl;
    std::cout << "Matrix size: " << N << "x" << N << std::endl;
    std::cout << "Memory per matrix: " << (N * N * sizeof(double) / (1024.0 * 1024.0)) << " MB" << std::endl;
    
    // Get GPU properties
    int deviceCount;
    cudaGetDeviceCount(&deviceCount);
    if (deviceCount == 0) {
        std::cerr << "No CUDA devices found!" << std::endl;
        return 1;
    }
    
    cudaDeviceProp prop;
    cudaGetDeviceProperties(&prop, 0);
    std::cout << "GPU: " << prop.name << std::endl;
    std::cout << "Compute Capability: " << prop.major << "." << prop.minor << std::endl;
    std::cout << "Global Memory: " << (prop.totalGlobalMem / (1024.0 * 1024.0 * 1024.0)) << " GB" << std::endl;
    
    // Allocate pinned memory on host
    double *h_A, *h_B, *h_C, *h_C_cpu;
    size_t matrixSize = N * N * sizeof(double);
    
    cudaMallocHost((void**)&h_A, matrixSize);
    cudaMallocHost((void**)&h_B, matrixSize);
    cudaMallocHost((void**)&h_C, matrixSize);
    h_C_cpu = new double[N * N];
    
    // Initialize matrices
    std::cout << "\nInitializing matrices..." << std::endl;
    initializeMatrix(h_A, N);
    initializeMatrix(h_B, N);
    
    // Allocate device memory
    double *d_A, *d_B, *d_C;
    cudaMalloc((void**)&d_A, matrixSize);
    cudaMalloc((void**)&d_B, matrixSize);
    cudaMalloc((void**)&d_C, matrixSize);
    
    // Copy matrices from host to device (pinned memory)
    std::cout << "Copying matrices to device (pinned memory)..." << std::endl;
    cudaMemcpy(d_A, h_A, matrixSize, cudaMemcpyHostToDevice);
    cudaMemcpy(d_B, h_B, matrixSize, cudaMemcpyHostToDevice);
    
    // Configure kernel launch parameters
    // Use 16x16 thread blocks
    int blockSize = 16;
    dim3 blockDim(blockSize, blockSize);
    dim3 gridDim((N + blockSize - 1) / blockSize, (N + blockSize - 1) / blockSize);
    
    std::cout << "Grid: (" << gridDim.x << ", " << gridDim.y << ")" << std::endl;
    std::cout << "Block: (" << blockDim.x << ", " << blockDim.y << ")" << std::endl;
    
    // Warm up
    matrixMultiplyGlobal<<<gridDim, blockDim>>>(d_A, d_B, d_C, N);
    cudaDeviceSynchronize();
    
    // Create CUDA events for timing
    cudaEvent_t start, stop;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);
    
    // Benchmark
    const int numRuns = 10;
    float totalTime = 0.0f;
    
    std::cout << "\nRunning " << numRuns << " iterations..." << std::endl;
    for (int run = 0; run < numRuns; ++run) {
        cudaEventRecord(start);
        matrixMultiplyGlobal<<<gridDim, blockDim>>>(d_A, d_B, d_C, N);
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
    std::ofstream csvFile("task1_results.csv", std::ios::app);
    bool fileExists = csvFile.tellp() > 0;
    if (!fileExists) {
        csvFile << "task,N,time_ms,gflops\n";
    }
    csvFile << "task1," << N << "," << std::fixed << std::setprecision(3) 
            << avgTime << "," << std::setprecision(2) << gflops << "\n";
    csvFile.close();
    std::cout << "Results saved to task1_results.csv" << std::endl;
    
    // Copy result back to host
    cudaMemcpy(h_C, d_C, matrixSize, cudaMemcpyDeviceToHost);
    
    // Verify results
    std::cout << "\nVerifying results..." << std::endl;
    matrixMultiplyCPU(h_A, h_B, h_C_cpu, N);
    verifyResults(h_C, h_C_cpu, N);
    
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
    
    std::cout << "\nDone!" << std::endl;
    return 0;
}

