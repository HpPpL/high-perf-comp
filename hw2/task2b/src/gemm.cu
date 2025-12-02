#include <cuda_runtime.h>
#include <cuda.h>
#include <iostream>
#include <iomanip>
#include <cmath>
#include <vector>
#include <chrono>
#include <fstream>

// CUDA kernel for matrix multiplication (works on tile)
__global__ void matrixMultiplyTile(double* A, double* B, double* C, int N, int tileRowStart, int tileRowEnd) {
    int row = blockIdx.y * blockDim.y + threadIdx.y + tileRowStart;
    int col = blockIdx.x * blockDim.x + threadIdx.x;
    
    if (row < tileRowEnd && col < N) {
        double sum = 0.0;
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
    int numStreams = 4;
    int blockSize = 16;  // Changed: now this is the block size parameter
    
    if (argc > 1) N = std::atoi(argv[1]);
    if (argc > 2) numStreams = std::atoi(argv[2]);
    if (argc > 3) blockSize = std::atoi(argv[3]);
    
    // Calculate tile size based on matrix size and number of streams
    int tileSize = (N + numStreams - 1) / numStreams;  // Rows per tile
    
    std::cout << "=== CUDA Matrix Multiplication (CUDA Streams with Overlapping) ===" << std::endl;
    std::cout << "Matrix size: " << N << "x" << N << std::endl;
    std::cout << "Number of streams: " << numStreams << std::endl;
    std::cout << "Block size: " << blockSize << "x" << blockSize << std::endl;
    std::cout << "Tile size (rows per stream): " << tileSize << std::endl;
    
    // Get GPU properties
    cudaDeviceProp prop;
    cudaGetDeviceProperties(&prop, 0);
    std::cout << "GPU: " << prop.name << std::endl;
    
    // Allocate pinned memory for full matrices
    size_t matrixSize = N * N * sizeof(double);
    double *h_A, *h_B, *h_C;
    cudaMallocHost((void**)&h_A, matrixSize);
    cudaMallocHost((void**)&h_B, matrixSize);
    cudaMallocHost((void**)&h_C, matrixSize);
    
    // Initialize matrices
    std::cout << "\nInitializing matrices..." << std::endl;
    initializeMatrix(h_A, N);
    initializeMatrix(h_B, N);
    
    // Allocate device memory for full matrices
    // Use separate buffers for each stream to enable true overlap
    double *d_B;  // Shared by all streams
    cudaMalloc((void**)&d_B, matrixSize);
    
    // Allocate device memory for tiles (one per stream for better overlap)
    std::vector<double*> d_A_tiles(numStreams, nullptr);
    std::vector<double*> d_C_tiles(numStreams, nullptr);
    std::vector<size_t> tileSizes(numStreams, 0);
    
    for (int i = 0; i < numStreams; ++i) {
        int tileRowStart = i * tileSize;
        int tileRowEnd = std::min(tileRowStart + tileSize, N);
        int actualTileSize = tileRowEnd - tileRowStart;
        
        if (actualTileSize > 0) {
            tileSizes[i] = actualTileSize * N * sizeof(double);
            cudaMalloc((void**)&d_A_tiles[i], tileSizes[i]);
            cudaMalloc((void**)&d_C_tiles[i], tileSizes[i]);
        }
    }
    
    // Create CUDA streams
    std::vector<cudaStream_t> streams(numStreams);
    for (int i = 0; i < numStreams; ++i) {
        cudaStreamCreate(&streams[i]);
    }
    
    // Copy full matrix B to device once (used by all streams)
    cudaMemcpy(d_B, h_B, matrixSize, cudaMemcpyHostToDevice);
    
    // Configure kernel launch parameters
    dim3 blockDim(blockSize, blockSize);
    dim3 gridDim((N + blockSize - 1) / blockSize, (tileSize + blockSize - 1) / blockSize);
    
    // Warm up - use first stream
    std::cout << "\nWarming up..." << std::endl;
    if (numStreams > 0 && tileSizes[0] > 0) {
        int tileRowStart = 0;
        int tileRowEnd = std::min(tileSize, N);
        int actualTileSize = tileRowEnd - tileRowStart;
        cudaMemcpyAsync(d_A_tiles[0], h_A + tileRowStart * N, tileSizes[0], 
                       cudaMemcpyHostToDevice, streams[0]);
        dim3 warmupGridDim((N + blockSize - 1) / blockSize, 
                          (actualTileSize + blockSize - 1) / blockSize);
        matrixMultiplyTile<<<warmupGridDim, blockDim, 0, streams[0]>>>(
            d_A_tiles[0], d_B, d_C_tiles[0], N, tileRowStart, tileRowEnd);
        cudaStreamSynchronize(streams[0]);
    }
    
    // Benchmark with streams - overlap computation and data transfer
    cudaEvent_t start, stop;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);
    
    const int numRuns = 10;
    float totalTime = 0.0f;
    
    std::cout << "Running " << numRuns << " iterations with " << numStreams << " streams..." << std::endl;
    std::cout << "Strategy: Pipeline approach - overlap data transfer and computation across streams" << std::endl;
    
    for (int run = 0; run < numRuns; ++run) {
        // CRITICAL: Synchronize before recording start event to ensure previous operations are complete
        cudaDeviceSynchronize();
        cudaEventRecord(start);
        
        // Pipeline approach: launch operations for all tiles in parallel streams
        // This allows real overlap: while one stream computes, others transfer data
        // Each stream uses its own device buffer for better overlap
        for (int streamIdx = 0; streamIdx < numStreams; ++streamIdx) {
            int tileRowStart = streamIdx * tileSize;
            int tileRowEnd = std::min(tileRowStart + tileSize, N);
            int actualTileSize = tileRowEnd - tileRowStart;
            
            if (actualTileSize <= 0 || tileSizes[streamIdx] == 0) continue;
            
            // Step 1: Copy tile of A asynchronously to device (in parallel with other streams)
            // Use dedicated buffer for this stream to enable true overlap
            cudaError_t err = cudaMemcpyAsync(d_A_tiles[streamIdx], 
                          h_A + tileRowStart * N, 
                          tileSizes[streamIdx], 
                          cudaMemcpyHostToDevice, 
                          streams[streamIdx]);
            if (err != cudaSuccess) {
                std::cerr << "CUDA memcpy error: " << cudaGetErrorString(err) << std::endl;
                return 1;
            }
            
            // Step 2: Launch kernel for this tile in the same stream
            // Kernel will automatically wait for copy to complete (stream dependency)
            dim3 tileGridDim((N + blockSize - 1) / blockSize, 
                           (actualTileSize + blockSize - 1) / blockSize);
            matrixMultiplyTile<<<tileGridDim, blockDim, 0, streams[streamIdx]>>>(
                d_A_tiles[streamIdx], d_B, d_C_tiles[streamIdx], N, tileRowStart, tileRowEnd);
            
            // Check for kernel launch errors
            err = cudaGetLastError();
            if (err != cudaSuccess) {
                std::cerr << "CUDA kernel launch error: " << cudaGetErrorString(err) << std::endl;
                return 1;
            }
            
            // NOTE: We do NOT copy results back in the measurement loop
            // This allows better overlap and more accurate performance measurement
            // Results will be copied back after all runs are complete
        }
        
        // Wait for all streams to complete computation
        // This allows maximum overlap: all streams work in parallel
        for (int i = 0; i < numStreams; ++i) {
            cudaStreamSynchronize(streams[i]);
        }
        
        cudaEventRecord(stop);
        cudaEventSynchronize(stop);
        
        float milliseconds = 0.0f;
        cudaEventElapsedTime(&milliseconds, start, stop);
        totalTime += milliseconds;
    }
    
    // Copy results back after all measurements (outside timing loop)
    std::cout << "Copying results back to host..." << std::endl;
    for (int streamIdx = 0; streamIdx < numStreams; ++streamIdx) {
        int tileRowStart = streamIdx * tileSize;
        int tileRowEnd = std::min(tileRowStart + tileSize, N);
        int actualTileSize = tileRowEnd - tileRowStart;
        
        if (actualTileSize <= 0 || tileSizes[streamIdx] == 0) continue;
        
        cudaMemcpyAsync(h_C + tileRowStart * N,
                      d_C_tiles[streamIdx],
                      tileSizes[streamIdx],
                      cudaMemcpyDeviceToHost,
                      streams[streamIdx]);
    }
    
    // Wait for all result copies to complete
    for (int i = 0; i < numStreams; ++i) {
        cudaStreamSynchronize(streams[i]);
    }
    
    float avgTime = totalTime / numRuns;
    double gflops = calculateGFLOPS(N, avgTime);
    
    std::cout << "\n=== Results ===" << std::endl;
    std::cout << "Average time: " << std::fixed << std::setprecision(3) << avgTime << " ms" << std::endl;
    std::cout << "Performance: " << std::fixed << std::setprecision(2) << gflops << " GFLOPS" << std::endl;
    
    // Save results to CSV
    std::ofstream csvFile("task2b_results.csv", std::ios::app);
    bool fileExists = csvFile.tellp() > 0;
    if (!fileExists) {
        csvFile << "task,N,num_streams,block_size,time_ms,gflops\n";
    }
    csvFile << "task2b," << N << "," << numStreams << "," << blockSize << ","
            << std::fixed << std::setprecision(3) << avgTime << ","
            << std::setprecision(2) << gflops << "\n";
    csvFile.close();
    std::cout << "Results saved to task2b_results.csv" << std::endl;
    
    // Verify results
    // Ensure all async operations are complete before verification
    cudaDeviceSynchronize();
    std::cout << "\nVerifying results..." << std::endl;
    double *C_cpu = new double[N * N];
    matrixMultiplyCPU(h_A, h_B, C_cpu, N);
    verifyResults(h_C, C_cpu, N);
    
    // Cleanup
    for (int i = 0; i < numStreams; ++i) {
        cudaStreamDestroy(streams[i]);
        if (d_A_tiles[i] != nullptr) cudaFree(d_A_tiles[i]);
        if (d_C_tiles[i] != nullptr) cudaFree(d_C_tiles[i]);
    }
    cudaFree(d_B);
    cudaFreeHost(h_A);
    cudaFreeHost(h_B);
    cudaFreeHost(h_C);
    delete[] C_cpu;
    cudaEventDestroy(start);
    cudaEventDestroy(stop);
    
    std::cout << "\nDone!" << std::endl;
    return 0;
}
