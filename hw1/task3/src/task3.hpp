#pragma once
#include "task3.h"
#ifdef _OPENMP
#include <omp.h>
#endif 
#include <iostream>
#include <iomanip>
#include <cmath>

// OpenBLAS includes
#ifdef USE_OPENBLAS
#include <cblas.h>
#endif

// MKL includes
#ifdef USE_MKL
#include <mkl.h>
#include <mkl_cblas.h>
#endif

namespace dgemm_benchmark {
    // Original sequential implementation
    void seq_multi(int N, double* A, double* B, double* C) {
        for (int i = 0; i < N; ++i) {
            for (int j = 0; j < N; ++j) {
                double sum = 0;
                for (int k = 0; k < N; ++k) {
                    sum += A[k * N + i] * B[j * N + k];
                }
                C[j * N + i] = sum;
            }
        }
    }
    
    // Original OpenMP implementation
    void omp_multi(int N, double* A, double* B, double* C){
        #pragma omp parallel for collapse(2)
        for (int i = 0; i < N; ++i) {
            for (int j = 0; j < N; ++j){
                double sum = 0.0;
                #pragma omp simd reduction(+:sum)
                for (int k = 0; k < N; ++k){
                    sum += A[k * N + i] * B[j * N + k];
                }
                C[j * N + i] = sum;
            }
        }
    }
    
    // OpenBLAS implementation
    void openblas_multi(int N, double* A, double* B, double* C) {
#ifdef USE_OPENBLAS
        // C = alpha * A * B + beta * C
        // For our case: C = 1.0 * A * B + 0.0 * C
        cblas_dgemm(CblasColMajor, CblasNoTrans, CblasNoTrans,
                    N, N, N, 1.0, A, N, B, N, 0.0, C, N);
#else
        std::cerr << "OpenBLAS not available! Please compile with -DUSE_OPENBLAS" << std::endl;
        // Fallback to sequential
        seq_multi(N, A, B, C);
#endif
    }
    
    // MKL implementation
    void mkl_multi(int N, double* A, double* B, double* C) {
#ifdef USE_MKL
        // C = alpha * A * B + beta * C
        // For our case: C = 1.0 * A * B + 0.0 * C
        cblas_dgemm(CblasColMajor, CblasNoTrans, CblasNoTrans,
                    N, N, N, 1.0, A, N, B, N, 0.0, C, N);
#else
        std::cerr << "MKL not available! Please compile with -DUSE_MKL" << std::endl;
        // Fallback to sequential
        seq_multi(N, A, B, C);
#endif
    }
    
    // Utility function to print matrix
    void print_matrix(int N, double* matrix, const std::string& name) {
        std::cout << name << " (" << N << "x" << N << "):" << std::endl;
        for (int i = 0; i < std::min(N, 5); ++i) {  // Print only first 5x5 for large matrices
            for (int j = 0; j < std::min(N, 5); ++j) {
                std::cout << std::setw(8) << std::fixed << std::setprecision(2) 
                         << matrix[j * N + i] << " ";
            }
            if (N > 5) std::cout << "...";
            std::cout << std::endl;
        }
        if (N > 5) std::cout << "..." << std::endl;
        std::cout << std::endl;
    }
    
    // Utility function to verify results
    bool verify_results(int N, double* C1, double* C2, double tolerance) {
        double max_error = 0.0;
        for (int i = 0; i < N * N; ++i) {
            double error = std::abs(C1[i] - C2[i]);
            max_error = std::max(max_error, error);
            if (error > tolerance) {
                return false;
            }
        }
        return true;
    }
    
    // Calculate GFLOPS
    double calculate_gflops(int N, double time_seconds) {
        // DGEMM: 2*N^3 floating point operations
        double operations = 2.0 * N * N * N;
        return operations / (time_seconds * 1e9); // Convert to GFLOPS
    }
}
