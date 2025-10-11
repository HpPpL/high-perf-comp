#pragma once

namespace dgemm {
    void seq_multi(int N, double* A, double* B, double* C);
    void omp_multi(int N, double* A, double* B, double* C);
    void print_matrix(int N, double* matrix, const std::string& name);
}
