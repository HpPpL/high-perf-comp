#include <cuda_runtime.h>
#include <cublas_v2.h>
#include <iostream>
#include <iomanip>
#include <cmath>
#include <vector>
#include <chrono>
#include <fstream>

// Calculate GFLOPS
double calculateGFLOPS(int N, double time_ms) {
    double operations = 2.0 * N * N * N;
    return operations / (time_ms * 1e6);
}

// Initialize matrix
void initializeMatrix(double* matrix, int N) {
    for (int i = 0; i < N; ++i) {
        for (int j = 0; j < N; ++j) {
            matrix[j * N + i] = (double)(i + j) / N;
        }
    }
}

// Verify results
bool verifyResults(double* C1, double* C2, int N, double tolerance = 1e-6) {
    double max_error = 0.0;
    for (int i = 0; i < N * N; ++i) {
        double error = std::abs(C1[i] - C2[i]);
        max_error = std::max(max_error, error);
        if (error > tolerance) {
            std::cout << "Mismatch at index " << i << ": C1=" << C1[i] 
                      << ", C2=" << C2[i] << ", error=" << error << std::endl;
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

int main(int argc, char* argv[]) {
    int N = 2048;
    if (argc > 1) {
        N = std::atoi(argv[1]);
    }
    
    std::cout << "=== cuBLAS Matrix Multiplication ===" << std::endl;
    std::cout << "Matrix size: " << N << "x" << N << std::endl;
    std::cout << "Memory per matrix: " << (N * N * sizeof(double) / (1024.0 * 1024.0)) << " MB" << std::endl;
    
    // Get GPU properties
    cudaDeviceProp prop;
    cudaGetDeviceProperties(&prop, 0);
    std::cout << "GPU: " << prop.name << std::endl;
    
    // Create cuBLAS handle
    cublasHandle_t handle;
    cublasCreate(&handle);
    
    // Allocate pinned memory
    size_t matrixSize = N * N * sizeof(double);
    double *h_A, *h_B, *h_C, *h_C_ref;
    cudaMallocHost((void**)&h_A, matrixSize);
    cudaMallocHost((void**)&h_B, matrixSize);
    cudaMallocHost((void**)&h_C, matrixSize);
    h_C_ref = new double[N * N];
    
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
    
    // cuBLAS uses column-major order
    // C = alpha * A * B + beta * C
    // For our case: C = 1.0 * A * B + 0.0 * C
    const double alpha = 1.0;
    const double beta = 0.0;
    
    // Warm up
    std::cout << "\nWarming up..." << std::endl;
    cublasDgemm(handle, CUBLAS_OP_N, CUBLAS_OP_N,
                N, N, N,
                &alpha,
                d_A, N,
                d_B, N,
                &beta,
                d_C, N);
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
        
        // cuBLAS DGEMM: C = alpha * A * B + beta * C
        cublasDgemm(handle, CUBLAS_OP_N, CUBLAS_OP_N,
                    N, N, N,
                    &alpha,
                    d_A, N,
                    d_B, N,
                    &beta,
                    d_C, N);
        
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
    std::ofstream csvFile("task4_results.csv", std::ios::app);
    bool fileExists = csvFile.tellp() > 0;
    if (!fileExists) {
        csvFile << "task,N,time_ms,gflops\n";
    }
    csvFile << "task4," << N << "," << std::fixed << std::setprecision(3) 
            << avgTime << "," << std::setprecision(2) << gflops << "\n";
    csvFile.close();
    std::cout << "Results saved to task4_results.csv" << std::endl;
    
    // Copy result back
    cudaMemcpy(h_C, d_C, matrixSize, cudaMemcpyDeviceToHost);
    
    // Verify results
    std::cout << "\nVerifying results..." << std::endl;
    matrixMultiplyCPU(h_A, h_B, h_C_ref, N);
    verifyResults(h_C, h_C_ref, N);
    
    // Cleanup
    cublasDestroy(handle);
    cudaFree(d_A);
    cudaFree(d_B);
    cudaFree(d_C);
    cudaFreeHost(h_A);
    cudaFreeHost(h_B);
    cudaFreeHost(h_C);
    delete[] h_C_ref;
    cudaEventDestroy(start);
    cudaEventDestroy(stop);
    
    std::cout << "\nDone!" << std::endl;
    return 0;
}

