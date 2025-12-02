#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <omp.h>
#include <time.h>

// Calculate GFLOPS
double calculateGFLOPS(int N, double time_ms) {
    double operations = 2.0 * N * N * N;
    return operations / (time_ms * 1e6);
}

// Initialize matrix
void initializeMatrix(double* matrix, int N) {
    #pragma omp target teams distribute parallel for collapse(2)
    for (int i = 0; i < N; ++i) {
        for (int j = 0; j < N; ++j) {
            matrix[j * N + i] = (double)(i + j) / N;
        }
    }
}

// OpenMP target version of matrix multiplication
void matrixMultiplyOpenMP(double* A, double* B, double* C, int N) {
    #pragma omp target teams distribute parallel for collapse(2) \
        map(to: A[0:N*N], B[0:N*N]) map(tofrom: C[0:N*N])
    for (int i = 0; i < N; ++i) {
        for (int j = 0; j < N; ++j) {
            double sum = 0.0;
            for (int k = 0; k < N; ++k) {
                // Column-major: A[k*N + i], B[j*N + k]
                sum += A[k * N + i] * B[j * N + k];
            }
            C[j * N + i] = sum;
        }
    }
}

// CPU reference
void matrixMultiplyCPU(double* A, double* B, double* C, int N) {
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

// Verify results
int verifyResults(double* C1, double* C2, int N, double tolerance) {
    double max_error = 0.0;
    for (int i = 0; i < N * N; ++i) {
        double error = fabs(C1[i] - C2[i]);
        max_error = (error > max_error) ? error : max_error;
        if (error > tolerance) {
            printf("Mismatch at index %d: C1=%.6f, C2=%.6f, error=%.6f\n", 
                   i, C1[i], C2[i], error);
            return 0;
        }
    }
    printf("Verification passed! Max error: %.6e\n", max_error);
    return 1;
}

int main(int argc, char* argv[]) {
    int N = 1024;
    if (argc > 1) {
        N = atoi(argv[1]);
    }
    
    printf("=== OpenMP GPU Matrix Multiplication ===\n");
    printf("Matrix size: %dx%d\n", N, N);
    printf("Memory per matrix: %.2f MB\n", (N * N * sizeof(double) / (1024.0 * 1024.0)));
    
    // Check OpenMP target support
    int num_devices = omp_get_num_devices();
    printf("Number of OpenMP devices: %d\n", num_devices);
    
    if (num_devices == 0) {
        printf("Warning: No OpenMP target devices found!\n");
        printf("Make sure to compile with: nvc -mp=gpu\n");
    }
    
    // Allocate memory
    size_t matrixSize = N * N * sizeof(double);
    double *A = (double*)malloc(matrixSize);
    double *B = (double*)malloc(matrixSize);
    double *C = (double*)malloc(matrixSize);
    double *C_ref = (double*)malloc(matrixSize);
    
    if (!A || !B || !C || !C_ref) {
        printf("Error: Memory allocation failed!\n");
        return 1;
    }
    
    // Initialize matrices on host
    printf("\nInitializing matrices on host...\n");
    for (int i = 0; i < N; ++i) {
        for (int j = 0; j < N; ++j) {
            A[j * N + i] = (double)(i + j) / N;
            B[j * N + i] = (double)(i * j) / (N * N);
        }
    }
    
    // Allocate on device (first touch)
    #pragma omp target enter data map(alloc: A[0:N*N], B[0:N*N], C[0:N*N])
    
    // Copy to device
    #pragma omp target update to(A[0:N*N], B[0:N*N])
    
    // Warm up
    printf("\nWarming up...\n");
    matrixMultiplyOpenMP(A, B, C, N);
    
    // Benchmark
    const int numRuns = 10;
    double totalTime = 0.0;
    
    printf("Running %d iterations...\n", numRuns);
    for (int run = 0; run < numRuns; ++run) {
        double start = omp_get_wtime();
        matrixMultiplyOpenMP(A, B, C, N);
        double end = omp_get_wtime();
        totalTime += (end - start) * 1000.0; // Convert to milliseconds
    }
    
    double avgTime = totalTime / numRuns;
    double gflops = calculateGFLOPS(N, avgTime);
    
    printf("\n=== Results ===\n");
    printf("Average time: %.3f ms\n", avgTime);
    printf("Performance: %.2f GFLOPS\n", gflops);
    
    // Copy result back
    #pragma omp target update from(C[0:N*N])
    
    // Verify results
    printf("\nVerifying results...\n");
    matrixMultiplyCPU(A, B, C_ref, N);
    verifyResults(C, C_ref, N, 1e-6);
    
    // Cleanup
    #pragma omp target exit data map(delete: A[0:N*N], B[0:N*N], C[0:N*N])
    free(A);
    free(B);
    free(C);
    free(C_ref);
    
    printf("\nDone!\n");
    return 0;
}

