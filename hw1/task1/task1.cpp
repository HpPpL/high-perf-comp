#include <iostream>
#include <chrono>
#include <cmath>
#include "task1.h"

using namespace dgemm;
using namespace std::chrono;

int main () {
    int N = 1000; 
    
    std::cout << "Matrix size: " << N << "x" << N << std::endl;
    std::cout << "Memory usage: " << (3 * N * N * sizeof(double) / (1024.0 * 1024.0)) << " MB" << std::endl;
    
    double *A = new double [N * N];
    double *B = new double [N * N];
    double *C_seq = new double [N * N];
    double *C_omp = new double [N * N];
    
    std::cout << "Initializing matrices..." << std::endl;
    for (int i = 0; i < N; ++i) {
        for (int j = 0; j < N; ++j) {
            A[j * N + i] = (double)(i + j) / N; // Column-major
            B[j * N + i] = (double)(i * j) / (N * N); // Column-major
        }
    }
    
    std::cout << "\nTesting sequential version..." << std::endl;
    auto start = high_resolution_clock::now();
    seq_multi(N, A, B, C_seq);
    auto end = high_resolution_clock::now();
    auto duration_seq = duration_cast<milliseconds>(end - start);
    
    std::cout << "Sequential time: " << duration_seq.count() << " ms" << std::endl;
    
    std::cout << "\nTesting OpenMP version..." << std::endl;
    start = high_resolution_clock::now();
    omp_multi(N, A, B, C_omp);
    end = high_resolution_clock::now();
    auto duration_omp = duration_cast<milliseconds>(end - start);
    
    std::cout << "OpenMP time: " << duration_omp.count() << " ms" << std::endl;
    
    std::cout << "\nVerifying results..." << std::endl;
    bool results_match = true;
    double max_error = 0.0;
    for (int i = 0; i < N * N; ++i) {
        double error = std::abs(C_seq[i] - C_omp[i]);
        max_error = std::max(max_error, error);
        if (error > 1e-10) {
            results_match = false;
            break;
        }
    }
    
    if (results_match) {
        std::cout << "✓ Results match! Max error: " << max_error << std::endl;
    } else {
        std::cout << "✗ Results don't match! Max error: " << max_error << std::endl;
    }
    
    if (duration_omp.count() > 0) {
        double speedup = (double)duration_seq.count() / duration_omp.count();
        std::cout << "Speedup: " << speedup << "x" << std::endl;
    }
    
    std::cout << "\nSample results (first few elements):" << std::endl;
    std::cout << "C[0][0] = " << C_seq[0] << std::endl;
    std::cout << "C[0][1] = " << C_seq[N] << std::endl;
    std::cout << "C[1][0] = " << C_seq[1] << std::endl;
    std::cout << "C[1][1] = " << C_seq[N + 1] << std::endl;
    
    delete[] A;
    delete[] B;
    delete[] C_seq;
    delete[] C_omp;
    
    return 0;
}