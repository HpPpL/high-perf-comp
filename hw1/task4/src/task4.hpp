#pragma once
#include "task4.h"
#ifdef _OPENMP
#include <omp.h>
#endif
#include <iostream>
#include <iomanip>
#include <cmath>
#include <algorithm>
#include <numeric>

namespace omp_matrix_mult {

    // Sequential matrix multiplication (baseline)
    void sequential_multiply(int N, double* A, double* B, double* C) {
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
    
    // Basic OpenMP parallel implementation
    void omp_parallel_multiply(int N, double* A, double* B, double* C) {
        #pragma omp parallel for
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
    
    // OpenMP with collapse directive for better load balancing
    void omp_parallel_collapse_multiply(int N, double* A, double* B, double* C) {
        #pragma omp parallel for collapse(2)
        for (int i = 0; i < N; ++i) {
            for (int j = 0; j < N; ++j) {
                double sum = 0.0;
                #pragma omp simd reduction(+:sum)
                for (int k = 0; k < N; ++k) {
                    sum += A[k * N + i] * B[j * N + k];
                }
                C[j * N + i] = sum;
            }
        }
    }
    
    // OpenMP with dynamic scheduling for better load balancing
    void omp_parallel_schedule_multiply(int N, double* A, double* B, double* C) {
        #pragma omp parallel for schedule(dynamic, 16)
        for (int i = 0; i < N; ++i) {
            for (int j = 0; j < N; ++j) {
                double sum = 0.0;
                #pragma omp simd reduction(+:sum)
                for (int k = 0; k < N; ++k) {
                    sum += A[k * N + i] * B[j * N + k];
                }
                C[j * N + i] = sum;
            }
        }
    }
    
    // Initialize matrices with test data
    void initialize_matrices(int N, double* A, double* B) {
        for (int i = 0; i < N; ++i) {
            for (int j = 0; j < N; ++j) {
                A[j * N + i] = (double)(i + j) / N;
                B[j * N + i] = (double)(i * j) / (N * N);
            }
        }
    }
    
    // Verify results between two matrices
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
        return operations / (time_seconds * 1e9);
    }
    
    // Print matrix (first 5x5 for large matrices)
    void print_matrix(int N, double* matrix, const std::string& name) {
        std::cout << name << " (" << N << "x" << N << "):" << std::endl;
        for (int i = 0; i < std::min(N, 5); ++i) {
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
    
    // Profile a function with timing
    ProfileResult profile_function(const std::string& name, 
                                   void (*func)(int, double*, double*, double*),
                                   int N, int num_threads,
                                   double* A, double* B, double* C) {
        ProfileResult result;
        result.function_name = name;
        result.matrix_size = N;
        result.num_threads = num_threads;
        result.thread_times_ms.resize(num_threads, 0.0);
        
        // Set number of threads
#ifdef _OPENMP
        omp_set_num_threads(num_threads);
#endif
        
        // Warm up
        func(N, A, B, C);
        
        // Profile with multiple runs
        const int num_runs = 5;
        double total_time = 0.0;
        
        for (int run = 0; run < num_runs; ++run) {
            auto start = std::chrono::high_resolution_clock::now();
            func(N, A, B, C);
            auto end = std::chrono::high_resolution_clock::now();
            
            auto duration = std::chrono::duration_cast<std::chrono::microseconds>(end - start);
            total_time += duration.count() / 1000.0; // Convert to milliseconds
        }
        
        result.execution_time_ms = total_time / num_runs;
        result.gflops = calculate_gflops(N, result.execution_time_ms / 1000.0);
        
        return result;
    }
    
    // Detailed profiling of OpenMP implementation
    DetailedProfile detailed_profile_omp_multiply(int N, double* A, double* B, double* C) {
        DetailedProfile profile;
        
        // Profile initialization
        auto init_start = std::chrono::high_resolution_clock::now();
        initialize_matrices(N, A, B);
        auto init_end = std::chrono::high_resolution_clock::now();
        auto init_duration = std::chrono::duration_cast<std::chrono::microseconds>(init_end - init_start);
        profile.initialization_time_ms = init_duration.count() / 1000.0;
        
        // Profile computation
        auto comp_start = std::chrono::high_resolution_clock::now();
        omp_parallel_collapse_multiply(N, A, B, C);
        auto comp_end = std::chrono::high_resolution_clock::now();
        auto comp_duration = std::chrono::duration_cast<std::chrono::microseconds>(comp_end - comp_start);
        profile.computation_time_ms = comp_duration.count() / 1000.0;
        
        // Profile verification
        double* C_ref = new double[N * N];
        auto verif_start = std::chrono::high_resolution_clock::now();
        sequential_multiply(N, A, B, C_ref);
        bool correct = verify_results(N, C, C_ref);
        (void)correct; // Suppress unused variable warning
        auto verif_end = std::chrono::high_resolution_clock::now();
        auto verif_duration = std::chrono::duration_cast<std::chrono::microseconds>(verif_end - verif_start);
        profile.verification_time_ms = verif_duration.count() / 1000.0;
        
        profile.total_time_ms = profile.initialization_time_ms + profile.computation_time_ms + profile.verification_time_ms;
        
        delete[] C_ref;
        
        return profile;
    }
    
    // Save profile results to CSV
    void save_profile_results(const std::vector<ProfileResult>& results, const std::string& filename) {
        std::ofstream file(filename);
        if (!file.is_open()) {
            std::cerr << "Error: Could not open file " << filename << std::endl;
            return;
        }
        
        file << "function_name,matrix_size,num_threads,execution_time_ms,gflops\n";
        for (const auto& result : results) {
            file << result.function_name << "," << result.matrix_size << "," << result.num_threads << "," 
                 << std::fixed << std::setprecision(2) << result.execution_time_ms << ","
                 << std::fixed << std::setprecision(2) << result.gflops << "\n";
        }
        file.close();
        std::cout << "Profile results saved to " << filename << std::endl;
    }
    
    // Print profile summary
    void print_profile_summary(const std::vector<ProfileResult>& results) {
        std::cout << "\n=== Profile Summary ===" << std::endl;
        std::cout << std::setw(20) << "Function" << std::setw(8) << "Size" << std::setw(8) << "Threads" 
                  << std::setw(15) << "Time (ms)" << std::setw(12) << "GFLOPS" << std::endl;
        std::cout << std::string(65, '-') << std::endl;
        
        for (const auto& result : results) {
            std::cout << std::setw(20) << result.function_name 
                      << std::setw(8) << result.matrix_size 
                      << std::setw(8) << result.num_threads
                      << std::setw(15) << std::fixed << std::setprecision(2) << result.execution_time_ms
                      << std::setw(12) << std::fixed << std::setprecision(2) << result.gflops << std::endl;
        }
    }
}
