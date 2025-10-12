#include <iostream>
#include <chrono>
#include <cmath>
#include <vector>
#include <fstream>
#include <iomanip>
#ifdef _OPENMP
#include <omp.h>
#endif
#include "task3.h"

using namespace dgemm;
using namespace std::chrono;

struct BenchmarkResult {
    std::string method;
    int N;
    int threads;
    double time_ms;
    double gflops;
};

void run_benchmark(const std::string& method_name, 
                  void (*multiply_func)(int, double*, double*, double*),
                  int N, int num_threads, 
                  double* A, double* B, double* C,
                  std::vector<BenchmarkResult>& results) {
    
    std::cout << "Testing " << method_name << " (N=" << N << ", threads=" << num_threads << ")..." << std::endl;
    
    // Set number of OpenMP threads
#ifdef _OPENMP
    omp_set_num_threads(num_threads);
#endif
    
    // Warm up
    multiply_func(N, A, B, C);
    
    // Benchmark
    const int num_runs = 3;
    double total_time = 0.0;
    
    for (int run = 0; run < num_runs; ++run) {
        auto start = high_resolution_clock::now();
        multiply_func(N, A, B, C);
        auto end = high_resolution_clock::now();
        
        auto duration = duration_cast<microseconds>(end - start);
        total_time += duration.count() / 1000.0; // Convert to milliseconds
    }
    
    double avg_time_ms = total_time / num_runs;
    double avg_time_seconds = avg_time_ms / 1000.0;
    double gflops = calculate_gflops(N, avg_time_seconds);
    
    BenchmarkResult result;
    result.method = method_name;
    result.N = N;
    result.threads = num_threads;
    result.time_ms = avg_time_ms;
    result.gflops = gflops;
    
    results.push_back(result);
    
    std::cout << "  Time: " << std::fixed << std::setprecision(2) << avg_time_ms << " ms" << std::endl;
    std::cout << "  GFLOPS: " << std::fixed << std::setprecision(2) << gflops << std::endl;
}

void save_results_to_csv(const std::vector<BenchmarkResult>& results, const std::string& filename) {
    std::ofstream file(filename);
    if (!file.is_open()) {
        std::cerr << "Error: Could not open file " << filename << std::endl;
        return;
    }
    
    file << "method,N,threads,time_ms,gflops\n";
    for (const auto& result : results) {
        file << result.method << "," << result.N << "," << result.threads << "," 
             << std::fixed << std::setprecision(2) << result.time_ms << ","
             << std::fixed << std::setprecision(2) << result.gflops << "\n";
    }
    file.close();
    std::cout << "Results saved to " << filename << std::endl;
}

void print_summary_table(const std::vector<BenchmarkResult>& results) {
    std::cout << "\n=== Summary Results ===" << std::endl;
    std::cout << std::setw(12) << "Method" << std::setw(6) << "N" << std::setw(8) << "Threads" 
              << std::setw(12) << "Time (ms)" << std::setw(12) << "GFLOPS" << std::endl;
    std::cout << std::string(52, '-') << std::endl;
    
    for (const auto& result : results) {
        std::cout << std::setw(12) << result.method 
                  << std::setw(6) << result.N 
                  << std::setw(8) << result.threads
                  << std::setw(12) << std::fixed << std::setprecision(2) << result.time_ms
                  << std::setw(12) << std::fixed << std::setprecision(2) << result.gflops << std::endl;
    }
}

int main(int argc, char* argv[]) {
    std::vector<int> matrix_sizes = {500, 1000, 1500};
    std::vector<int> thread_counts = {1, 2, 4, 8, 16, 24};
    std::vector<BenchmarkResult> results;
    
    std::cout << "=== DGEMM Performance Comparison ===" << std::endl;
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
    double *C_omp = new double[test_N * test_N];
    double *C_openblas = new double[test_N * test_N];
    double *C_mkl = new double[test_N * test_N];
    
    // Initialize test matrices
    for (int i = 0; i < test_N; ++i) {
        for (int j = 0; j < test_N; ++j) {
            A[j * test_N + i] = (double)(i + j) / test_N;
            B[j * test_N + i] = (double)(i * j) / (test_N * test_N);
        }
    }
    
    // Run all methods
    seq_multi(test_N, A, B, C_seq);
    omp_multi(test_N, A, B, C_omp);
    openblas_multi(test_N, A, B, C_openblas);
    mkl_multi(test_N, A, B, C_mkl);
    
    // Verify results
    bool omp_correct = verify_results(test_N, C_seq, C_omp);
    bool openblas_correct = verify_results(test_N, C_seq, C_openblas);
    bool mkl_correct = verify_results(test_N, C_seq, C_mkl);
    
    std::cout << "Sequential vs OpenMP: " << (omp_correct ? "✓ PASS" : "✗ FAIL") << std::endl;
    std::cout << "Sequential vs OpenBLAS: " << (openblas_correct ? "✓ PASS" : "✗ FAIL") << std::endl;
    std::cout << "Sequential vs MKL: " << (mkl_correct ? "✓ PASS" : "✗ FAIL") << std::endl;
    
    delete[] A;
    delete[] B;
    delete[] C_seq;
    delete[] C_omp;
    delete[] C_openblas;
    delete[] C_mkl;
    
    if (!omp_correct || !openblas_correct || !mkl_correct) {
        std::cerr << "Correctness test failed! Aborting benchmarks." << std::endl;
        return 1;
    }
    
    std::cout << "\n=== Performance Benchmarks ===" << std::endl;
    
    // Run benchmarks
    for (int N : matrix_sizes) {
        std::cout << "\n=== Matrix size: " << N << "x" << N << " ===" << std::endl;
        
        // Allocate matrices for this size
        double *A = new double[N * N];
        double *B = new double[N * N];
        double *C = new double[N * N];
        
        // Initialize matrices
        for (int i = 0; i < N; ++i) {
            for (int j = 0; j < N; ++j) {
                A[j * N + i] = (double)(i + j) / N;
                B[j * N + i] = (double)(i * j) / (N * N);
            }
        }
        
        for (int P : thread_counts) {
            run_benchmark("Sequential", seq_multi, N, P, A, B, C, results);
            run_benchmark("OpenMP", omp_multi, N, P, A, B, C, results);
            run_benchmark("OpenBLAS", openblas_multi, N, P, A, B, C, results);
            run_benchmark("MKL", mkl_multi, N, P, A, B, C, results);
        }
        
        delete[] A;
        delete[] B;
        delete[] C;
    }
    
    // Save and display results
    save_results_to_csv(results, "blas_comparison_results.csv");
    print_summary_table(results);
    
    return 0;
}
