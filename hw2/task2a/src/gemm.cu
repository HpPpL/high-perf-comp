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

// CPU reference implementation
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
    int N = 1024;
    if (argc > 1) {
        N = std::atoi(argv[1]);
    }
    
    std::cout << "=== CUDA Matrix Multiplication (Unified Memory) ===" << std::endl;
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
    
    // Check unified memory support
    if (prop.major < 6) {
        std::cerr << "Warning: Unified Memory requires compute capability 6.0+" << std::endl;
    }
    
    // Allocate unified memory (no explicit transfer needed)
    double *A, *B, *C;
    size_t matrixSize = N * N * sizeof(double);
    
    std::cout << "\nAllocating unified memory..." << std::endl;
    cudaMallocManaged(&A, matrixSize);
    cudaMallocManaged(&B, matrixSize);
    cudaMallocManaged(&C, matrixSize);
    
    // Initialize matrices on CPU (unified memory is accessible from CPU)
    std::cout << "Initializing matrices on CPU..." << std::endl;
    initializeMatrix(A, N);
    initializeMatrix(B, N);
    
    // Configure kernel launch parameters
    int blockSize = 16;
    dim3 blockDim(blockSize, blockSize);
    dim3 gridDim((N + blockSize - 1) / blockSize, (N + blockSize - 1) / blockSize);
    
    std::cout << "Grid: (" << gridDim.x << ", " << gridDim.y << ")" << std::endl;
    std::cout << "Block: (" << blockDim.x << ", " << blockDim.y << ")" << std::endl;
    
    // Warm up
    std::cout << "\nWarming up..." << std::endl;
    matrixMultiplyGlobal<<<gridDim, blockDim>>>(A, B, C, N);
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
        cudaEventRecord(start);
        matrixMultiplyGlobal<<<gridDim, blockDim>>>(A, B, C, N);
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
    std::ofstream csvFile("task2a_results.csv", std::ios::app);
    bool fileExists = csvFile.tellp() > 0;
    if (!fileExists) {
        csvFile << "task,N,time_ms,gflops\n";
    }
    csvFile << "task2a," << N << "," << std::fixed << std::setprecision(3) 
            << avgTime << "," << std::setprecision(2) << gflops << "\n";
    csvFile.close();
    std::cout << "Results saved to task2a_results.csv" << std::endl;
    
    // Verify results (unified memory accessible from CPU)
    std::cout << "\nVerifying results..." << std::endl;
    double *C_cpu = new double[N * N];
    matrixMultiplyCPU(A, B, C_cpu, N);
    verifyResults(C, C_cpu, N);
    
    // Cleanup
    cudaFree(A);
    cudaFree(B);
    cudaFree(C);
    delete[] C_cpu;
    cudaEventDestroy(start);
    cudaEventDestroy(stop);
    
    std::cout << "\nDone!" << std::endl;
    return 0;
}

