#pragma once
#include <string>
#include <vector>
#include <chrono>

namespace omp_matrix_mult {
    // Matrix multiplication implementations
    void sequential_multiply(int N, double* A, double* B, double* C);
    void omp_parallel_multiply(int N, double* A, double* B, double* C);
    void omp_parallel_collapse_multiply(int N, double* A, double* B, double* C);
    void omp_parallel_schedule_multiply(int N, double* A, double* B, double* C);
    
    // Profiling structures
    struct ProfileResult {
        std::string function_name;
        int matrix_size;
        int num_threads;
        double execution_time_ms;
        double gflops;
        std::vector<double> thread_times_ms;
    };
    
    struct DetailedProfile {
        double initialization_time_ms;
        double computation_time_ms;
        double verification_time_ms;
        double total_time_ms;
        std::vector<double> per_thread_computation_ms;
    };
    
    // Profiling functions
    ProfileResult profile_function(const std::string& name, 
                                   void (*func)(int, double*, double*, double*),
                                   int N, int num_threads,
                                   double* A, double* B, double* C);
    
    DetailedProfile detailed_profile_omp_multiply(int N, double* A, double* B, double* C);
    
    // Utility functions
    void initialize_matrices(int N, double* A, double* B);
    bool verify_results(int N, double* C1, double* C2, double tolerance = 1e-10);
    double calculate_gflops(int N, double time_seconds);
    void print_matrix(int N, double* matrix, const std::string& name = "Matrix");
    void save_profile_results(const std::vector<ProfileResult>& results, const std::string& filename);
    void print_profile_summary(const std::vector<ProfileResult>& results);
}

#include "task4.hpp"
