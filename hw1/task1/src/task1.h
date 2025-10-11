#pragma once
#include <string>

namespace dgemm {
    void omp_multi(int N, double* A, double* B, double* C);
    void seq_multi(int N, double* A, double* B, double* C);
    void print_matrix(int N, double* matrix, const std::string& name = "Matrix");
}

#include "task1.hpp"

