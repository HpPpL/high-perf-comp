#include <iostream>
#include <vector>
#include <fstream>
#include <iomanip>
#ifdef _OPENMP
#include <omp.h>
#endif
#include "task4.h"

using namespace omp_matrix_mult;

int main(int argc, char* argv[]) {
    (void)argc; // Suppress unused parameter warning
    (void)argv; // Suppress unused parameter warning
    std::cout << "=== OpenMP Matrix Multiplication with Profiling ===" << std::endl;
    
    // Configuration
    std::vector<int> matrix_sizes = {500, 1000, 1500};
    std::vector<int> thread_counts = {1, 2, 4, 8, 16, 24};
    std::vector<ProfileResult> all_results;
    
    std::cout << "Matrix sizes: ";
    for (int N : matrix_sizes) {
        std::cout << N << " ";
    }
    std::cout << std::endl;
    std::cout << "Thread counts: ";
    for (int P : thread_counts) {
        std::cout << P << " ";
    }
    std::cout << std::endl << std::endl;
    
    // Test correctness first
    std::cout << "=== Correctness Verification ===" << std::endl;
    int test_N = 100;
    double *A = new double[test_N * test_N];
    double *B = new double[test_N * test_N];
    double *C_seq = new double[test_N * test_N];
    double *C_omp1 = new double[test_N * test_N];
    double *C_omp2 = new double[test_N * test_N];
    double *C_omp3 = new double[test_N * test_N];
    
    initialize_matrices(test_N, A, B);
    
    // Run all implementations
    sequential_multiply(test_N, A, B, C_seq);
    omp_parallel_multiply(test_N, A, B, C_omp1);
    omp_parallel_collapse_multiply(test_N, A, B, C_omp2);
    omp_parallel_schedule_multiply(test_N, A, B, C_omp3);
    
    // Verify results
    bool omp1_correct = verify_results(test_N, C_seq, C_omp1);
    bool omp2_correct = verify_results(test_N, C_seq, C_omp2);
    bool omp3_correct = verify_results(test_N, C_seq, C_omp3);
    
    std::cout << "Sequential vs OpenMP Basic: " << (omp1_correct ? "✓ PASS" : "✗ FAIL") << std::endl;
    std::cout << "Sequential vs OpenMP Collapse: " << (omp2_correct ? "✓ PASS" : "✗ FAIL") << std::endl;
    std::cout << "Sequential vs OpenMP Schedule: " << (omp3_correct ? "✓ PASS" : "✗ FAIL") << std::endl;
    
    delete[] A;
    delete[] B;
    delete[] C_seq;
    delete[] C_omp1;
    delete[] C_omp2;
    delete[] C_omp3;
    
    if (!omp1_correct || !omp2_correct || !omp3_correct) {
        std::cerr << "Correctness test failed! Aborting benchmarks." << std::endl;
        return 1;
    }
    
    std::cout << "\n=== Performance Profiling ===" << std::endl;
    
    // Run performance tests
    for (int N : matrix_sizes) {
        std::cout << "\n=== Matrix size: " << N << "x" << N << " ===" << std::endl;
        
        // Allocate matrices
        double *A = new double[N * N];
        double *B = new double[N * N];
        double *C = new double[N * N];
        
        initialize_matrices(N, A, B);
        
        for (int P : thread_counts) {
            std::cout << "\n--- Testing with " << P << " threads ---" << std::endl;
            
            // Profile each implementation
            ProfileResult seq_result = profile_function("Sequential", sequential_multiply, N, P, A, B, C);
            ProfileResult omp1_result = profile_function("OpenMP_Basic", omp_parallel_multiply, N, P, A, B, C);
            ProfileResult omp2_result = profile_function("OpenMP_Collapse", omp_parallel_collapse_multiply, N, P, A, B, C);
            ProfileResult omp3_result = profile_function("OpenMP_Schedule", omp_parallel_schedule_multiply, N, P, A, B, C);
            
            // Store results
            all_results.push_back(seq_result);
            all_results.push_back(omp1_result);
            all_results.push_back(omp2_result);
            all_results.push_back(omp3_result);
            
            // Print immediate results
            std::cout << "Sequential: " << std::fixed << std::setprecision(2) 
                      << seq_result.execution_time_ms << " ms, " 
                      << seq_result.gflops << " GFLOPS" << std::endl;
            std::cout << "OpenMP Basic: " << std::fixed << std::setprecision(2) 
                      << omp1_result.execution_time_ms << " ms, " 
                      << omp1_result.gflops << " GFLOPS" << std::endl;
            std::cout << "OpenMP Collapse: " << std::fixed << std::setprecision(2) 
                      << omp2_result.execution_time_ms << " ms, " 
                      << omp2_result.gflops << " GFLOPS" << std::endl;
            std::cout << "OpenMP Schedule: " << std::fixed << std::setprecision(2) 
                      << omp3_result.execution_time_ms << " ms, " 
                      << omp3_result.gflops << " GFLOPS" << std::endl;
            
            // Calculate speedup
            if (P > 1) {
                double speedup1 = seq_result.execution_time_ms / omp1_result.execution_time_ms;
                double speedup2 = seq_result.execution_time_ms / omp2_result.execution_time_ms;
                double speedup3 = seq_result.execution_time_ms / omp3_result.execution_time_ms;
                
                std::cout << "Speedup vs Sequential: Basic=" << std::fixed << std::setprecision(2) << speedup1
                          << ", Collapse=" << speedup2 << ", Schedule=" << speedup3 << std::endl;
            }
        }
        
        // Detailed profiling for the best OpenMP implementation
        std::cout << "\n--- Detailed Profiling (OpenMP Collapse) ---" << std::endl;
        DetailedProfile detailed = detailed_profile_omp_multiply(N, A, B, C);
        std::cout << "Initialization: " << std::fixed << std::setprecision(2) 
                  << detailed.initialization_time_ms << " ms" << std::endl;
        std::cout << "Computation: " << std::fixed << std::setprecision(2) 
                  << detailed.computation_time_ms << " ms" << std::endl;
        std::cout << "Verification: " << std::fixed << std::setprecision(2) 
                  << detailed.verification_time_ms << " ms" << std::endl;
        std::cout << "Total: " << std::fixed << std::setprecision(2) 
                  << detailed.total_time_ms << " ms" << std::endl;
        
        delete[] A;
        delete[] B;
        delete[] C;
    }
    
    // Save and display final results
    save_profile_results(all_results, "omp_profiling_results.csv");
    print_profile_summary(all_results);
    
    // Print OpenMP information
#ifdef _OPENMP
    std::cout << "\n=== OpenMP Information ===" << std::endl;
    std::cout << "OpenMP Version: " << _OPENMP << std::endl;
    std::cout << "Max threads available: " << omp_get_max_threads() << std::endl;
    std::cout << "Number of processors: " << omp_get_num_procs() << std::endl;
#endif
    
    return 0;
}
