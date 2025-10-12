#pragma once
#include <string>

namespace dgemm_benchmark {
    // Original implementations
    void seq_multi(int N, double* A, double* B, double* C);
    void omp_multi(int N, double* A, double* B, double* C);
    
    // BLAS implementations
    void openblas_multi(int N, double* A, double* B, double* C);
    void mkl_multi(int N, double* A, double* B, double* C);
    
    // Utility functions
    void print_matrix(int N, double* matrix, const std::string& name = "Matrix");
    bool verify_results(int N, double* C1, double* C2, double tolerance = 1e-10);
    double calculate_gflops(int N, double time_seconds);
}

#include "task3.hpp"
