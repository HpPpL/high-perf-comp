#include <iostream>
#include <chrono>
#include <cmath>
#include <vector>
#include <fstream>
#include <iomanip>
#include <omp.h>
#include "task2.h"
#include "task2.hpp"

using namespace dgemm;
using namespace std::chrono;

struct BenchmarkResult {
    int N;
    int threads;
    double time_ms;
    double gflops;
};

double calculate_gflops(int N, double time_seconds) {
    // DGEMM: 2*N^3 floating point operations
    double operations = 2.0 * N * N * N;
    return operations / (time_seconds * 1e9); // Convert to GFLOPS
}

void run_benchmark(int N, int num_threads, std::vector<BenchmarkResult>& results) {
    std::cout << "Running benchmark: N=" << N << ", threads=" << num_threads << std::endl;
    
    // Set number of OpenMP threads
    omp_set_num_threads(num_threads);
    
    // Allocate matrices
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
    
    // Warm up
    omp_multi(N, A, B, C);
    
    // Benchmark
    const int num_runs = 3;
    double total_time = 0.0;
    
    for (int run = 0; run < num_runs; ++run) {
        auto start = high_resolution_clock::now();
        omp_multi(N, A, B, C);
        auto end = high_resolution_clock::now();
        
        auto duration = duration_cast<microseconds>(end - start);
        total_time += duration.count() / 1000.0; // Convert to milliseconds
    }
    
    double avg_time_ms = total_time / num_runs;
    double avg_time_seconds = avg_time_ms / 1000.0;
    double gflops = calculate_gflops(N, avg_time_seconds);
    
    BenchmarkResult result;
    result.N = N;
    result.threads = num_threads;
    result.time_ms = avg_time_ms;
    result.gflops = gflops;
    
    results.push_back(result);
    
    std::cout << "  Time: " << std::fixed << std::setprecision(2) << avg_time_ms << " ms" << std::endl;
    std::cout << "  GFLOPS: " << std::fixed << std::setprecision(2) << gflops << std::endl;
    
    delete[] A;
    delete[] B;
    delete[] C;
}

void save_results_to_csv(const std::vector<BenchmarkResult>& results, const std::string& filename) {
    std::ofstream file(filename);
    if (!file.is_open()) {
        std::cerr << "Error: Could not open file " << filename << std::endl;
        return;
    }
    
    file << "N,threads,time_ms,gflops\n";
    for (const auto& result : results) {
        file << result.N << "," << result.threads << "," 
             << std::fixed << std::setprecision(2) << result.time_ms << ","
             << std::fixed << std::setprecision(2) << result.gflops << "\n";
    }
    file.close();
    std::cout << "Results saved to " << filename << std::endl;
}

int main(int argc, char* argv[]) {
    std::vector<int> matrix_sizes = {500, 1000, 1500};
    std::vector<int> thread_counts = {1, 2, 4, 8, 16, 24};
    std::vector<BenchmarkResult> results;
    
    std::cout << "=== Scalability Analysis for DGEMM ===" << std::endl;
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
    
    // Run benchmarks
    for (int N : matrix_sizes) {
        std::cout << "\n=== Matrix size: " << N << "x" << N << " ===" << std::endl;
        for (int P : thread_counts) {
            run_benchmark(N, P, results);
        }
    }
    
    // Save results
    save_results_to_csv(results, "scalability_results.csv");
    
    // Print summary table
    std::cout << "\n=== Summary Results ===" << std::endl;
    std::cout << std::setw(6) << "N" << std::setw(8) << "Threads" 
              << std::setw(12) << "Time (ms)" << std::setw(12) << "GFLOPS" << std::endl;
    std::cout << std::string(40, '-') << std::endl;
    
    for (const auto& result : results) {
        std::cout << std::setw(6) << result.N 
                  << std::setw(8) << result.threads
                  << std::setw(12) << std::fixed << std::setprecision(2) << result.time_ms
                  << std::setw(12) << std::fixed << std::setprecision(2) << result.gflops << std::endl;
    }
    
    return 0;
}
