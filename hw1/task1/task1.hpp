#pragma once
#include "task1.h"
#include <omp.h> 
#include <iostream>
#include <iomanip>

namespace dgemm {
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
    
    void print_matrix(int N, double* matrix, const std::string& name) {
        std::cout << name << " (" << N << "x" << N << "):" << std::endl;
        for (int i = 0; i < N; ++i) {
            for (int j = 0; j < N; ++j) {
                std::cout << std::setw(8) << std::fixed << std::setprecision(2) 
                         << matrix[j * N + i] << " ";
            }
            std::cout << std::endl;
        }
        std::cout << std::endl;
    }
    
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
}
