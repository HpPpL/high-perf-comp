#include <mpi.h>
#include <cuda_runtime.h>
#include <iostream>
#include <iomanip>
#include <cmath>
#include <vector>
#include <fstream>
#include <chrono>

// CUDA kernel для вычисления одного временного шага
__global__ void heat_step_kernel(double* u_curr, double* u_next, int local_N, double coeff) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    
    // Индексы смещены на 1 из-за граничных точек
    if (idx >= 1 && idx <= local_N) {
        u_next[idx] = u_curr[idx] + coeff * (u_curr[idx+1] - 2.0 * u_curr[idx] + u_curr[idx-1]);
    }
}

int main(int argc, char* argv[]) {
    MPI_Init(&argc, &argv);
    
    int rank, size;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &size);
    
    // Для гибридной задачи используем 2 процесса
    if (size != 2) {
        if (rank == 0) {
            std::cerr << "Эта программа требует ровно 2 MPI процесса!" << std::endl;
        }
        MPI_Abort(MPI_COMM_WORLD, 1);
    }
    
    // Параметры задачи
    const double k = 1.0;
    double h;
    const double l = 1.0;
    const double u0 = 1.0;
    const double T = 1e-4;
    
    int N = 10000;
    if (argc > 1) {
        N = std::atoi(argv[1]);
    }
    
    h = l / (N - 1);
    double tau = 0.9 * h * h / k;
    int num_time_steps = static_cast<int>(T / tau);
    if (num_time_steps == 0) num_time_steps = 1;
    
    if (rank == 0) {
        std::cout << "=== Task 5: Гибридная задача (MPI + GPU) ===" << std::endl;
        std::cout << "Параметры:" << std::endl;
        std::cout << "  N = " << N << " точек" << std::endl;
        std::cout << "  h = " << h << std::endl;
        std::cout << "  τ = " << tau << std::endl;
        std::cout << "  T = " << T << std::endl;
        std::cout << "  Количество шагов по времени: " << num_time_steps << std::endl;
        std::cout << "  Количество процессов: " << size << std::endl;
    }
    
    // Выбор GPU для каждого процесса
    int device_id = rank;
    cudaSetDevice(device_id);
    
    cudaDeviceProp prop;
    cudaGetDeviceProperties(&prop, device_id);
    std::cout << "Process " << rank << " using GPU: " << prop.name << std::endl;
    
    // Распределение точек между процессами
    int points_per_proc = N / size;
    int remainder = N % size;
    
    int local_start = rank * points_per_proc + std::min(rank, remainder);
    int local_end = local_start + points_per_proc + (rank < remainder ? 1 : 0);
    int local_N = local_end - local_start;
    
    // Выделение памяти на GPU
    double *d_u_curr, *d_u_next;
    size_t mem_size = (local_N + 2) * sizeof(double);
    
    cudaMalloc(&d_u_curr, mem_size);
    cudaMalloc(&d_u_next, mem_size);
    
    // Выделение памяти на хосте для начальных данных и обмена
    std::vector<double> h_u_initial(local_N);
    double *h_boundary_left = nullptr, *h_boundary_right = nullptr;
    double *h_boundary_send_right = nullptr;  // Отдельный буфер для отправки правой границы
    
    if (rank > 0) {
        cudaMallocHost(&h_boundary_left, sizeof(double));
    }
    if (rank < size - 1) {
        cudaMallocHost(&h_boundary_right, sizeof(double));
        cudaMallocHost(&h_boundary_send_right, sizeof(double));  // Отдельный буфер для отправки
    }
    
    // Инициализация начального распределения на root процессе
    std::vector<double> u_initial(N);
    if (rank == 0) {
        for (int i = 0; i < N; ++i) {
            double x = i * h;
            if (x >= 0.0 && x <= l) {
                u_initial[i] = u0;
            } else {
                u_initial[i] = 0.0;
            }
        }
        u_initial[0] = 0.0;
        u_initial[N-1] = 0.0;
    }
    
    // Распределение начальных данных
    std::vector<int> sendcounts(size);
    std::vector<int> displs(size);
    
    for (int i = 0; i < size; ++i) {
        int proc_start = i * points_per_proc + std::min(i, remainder);
        int proc_end = proc_start + points_per_proc + (i < remainder ? 1 : 0);
        sendcounts[i] = proc_end - proc_start;
        displs[i] = proc_start;
    }
    
    MPI_Scatterv(u_initial.data(), sendcounts.data(), displs.data(), MPI_DOUBLE,
                 h_u_initial.data(), local_N, MPI_DOUBLE, 0, MPI_COMM_WORLD);
    
    // Копирование начальных данных на GPU с учетом граничных точек
    std::vector<double> h_u_curr(local_N + 2);
    h_u_curr[0] = (rank == 0) ? 0.0 : 0.0;  // левая граница
    for (int i = 0; i < local_N; ++i) {
        h_u_curr[i + 1] = h_u_initial[i];
    }
    h_u_curr[local_N + 1] = (rank == size - 1) ? 0.0 : 0.0;  // правая граница
    
    cudaMemcpy(d_u_curr, h_u_curr.data(), mem_size, cudaMemcpyHostToDevice);
    cudaMemcpy(d_u_next, h_u_curr.data(), mem_size, cudaMemcpyHostToDevice);
    
    // Настройка CUDA kernel
    int threads_per_block = 256;
    int blocks = (local_N + threads_per_block - 1) / threads_per_block;
    double coeff = k * tau / (h * h);
    
    MPI_Barrier(MPI_COMM_WORLD);
    auto start_time = std::chrono::high_resolution_clock::now();
    
    // Основной цикл по времени
    for (int n = 0; n < num_time_steps; ++n) {
        // Синхронизация GPU перед обменом данными
        cudaDeviceSynchronize();
        
        // Обмен граничными значениями между процессами
        // Используем неблокирующие операции для избежания deadlock
        MPI_Request req_send_left, req_recv_left, req_send_right, req_recv_right;
        
        if (rank > 0) {
            // Копирование левой границы с GPU на хост (для отправки предыдущему процессу)
            cudaMemcpy(h_boundary_left, &d_u_curr[1], sizeof(double), cudaMemcpyDeviceToHost);
            cudaError_t err = cudaGetLastError();
            if (err != cudaSuccess) {
                std::cerr << "Process " << rank << " CUDA error (left boundary copy): " 
                          << cudaGetErrorString(err) << std::endl;
            }
            // Отправка предыдущему процессу (неблокирующая)
            MPI_Isend(h_boundary_left, 1, MPI_DOUBLE, rank - 1, 0, MPI_COMM_WORLD, &req_send_left);
            // Получение от предыдущего процесса (неблокирующая)
            MPI_Irecv(h_boundary_left, 1, MPI_DOUBLE, rank - 1, 0, MPI_COMM_WORLD, &req_recv_left);
        }
        
        if (rank < size - 1) {
            // Копирование правой границы с GPU на хост (для отправки следующему процессу)
            // Используем отдельный буфер, чтобы не перезаписать данные для приема
            cudaMemcpy(h_boundary_send_right, &d_u_curr[local_N], sizeof(double), cudaMemcpyDeviceToHost);
            cudaError_t err = cudaGetLastError();
            if (err != cudaSuccess) {
                std::cerr << "Process " << rank << " CUDA error (right boundary copy): " 
                          << cudaGetErrorString(err) << std::endl;
            }
            // Получение от следующего процесса (неблокирующая)
            MPI_Irecv(h_boundary_right, 1, MPI_DOUBLE, rank + 1, 0, MPI_COMM_WORLD, &req_recv_right);
            // Отправка следующему процессу (неблокирующая) - используем отдельный буфер
            MPI_Isend(h_boundary_send_right, 1, MPI_DOUBLE, rank + 1, 0, MPI_COMM_WORLD, &req_send_right);
        }
        
        // Ждем завершения получения граничных значений (необходимо для вычислений)
        if (rank > 0) {
            MPI_Wait(&req_recv_left, MPI_STATUS_IGNORE);
            // Копирование полученного значения обратно на GPU
            cudaMemcpy(&d_u_curr[0], h_boundary_left, sizeof(double), cudaMemcpyHostToDevice);
            cudaError_t err = cudaGetLastError();
            if (err != cudaSuccess) {
                std::cerr << "Process " << rank << " CUDA error (left boundary set): " 
                          << cudaGetErrorString(err) << std::endl;
            }
        }
        
        if (rank < size - 1) {
            MPI_Wait(&req_recv_right, MPI_STATUS_IGNORE);
            // Копирование полученного значения на GPU
            cudaMemcpy(&d_u_curr[local_N + 1], h_boundary_right, sizeof(double), cudaMemcpyHostToDevice);
            cudaError_t err = cudaGetLastError();
            if (err != cudaSuccess) {
                std::cerr << "Process " << rank << " CUDA error (right boundary set): " 
                          << cudaGetErrorString(err) << std::endl;
            }
        }
        
        // Вычисление нового временного слоя на GPU
        heat_step_kernel<<<blocks, threads_per_block>>>(d_u_curr, d_u_next, local_N, coeff);
        cudaError_t err = cudaGetLastError();
        if (err != cudaSuccess) {
            std::cerr << "Process " << rank << " CUDA kernel error: " 
                      << cudaGetErrorString(err) << std::endl;
        }
        
        // Ожидание завершения отправки (для синхронизации, но вычисления уже выполнены)
        if (rank > 0) {
            MPI_Wait(&req_send_left, MPI_STATUS_IGNORE);
        }
        
        if (rank < size - 1) {
            MPI_Wait(&req_send_right, MPI_STATUS_IGNORE);
        }
        
        cudaDeviceSynchronize();
        
        // Применение граничных условий
        if (rank == 0) {
            cudaMemset(&d_u_next[0], 0, sizeof(double));
        }
        if (rank == size - 1) {
            cudaMemset(&d_u_next[local_N + 1], 0, sizeof(double));
        }
        
        // Обмен указателями
        std::swap(d_u_curr, d_u_next);
    }
    
    cudaDeviceSynchronize();
    MPI_Barrier(MPI_COMM_WORLD);
    auto end_time = std::chrono::high_resolution_clock::now();
    
    // Копирование результатов обратно на хост
    cudaMemcpy(h_u_curr.data(), d_u_curr, mem_size, cudaMemcpyDeviceToHost);
    
    // Сбор результатов на root процессе
    std::vector<double> u_final(N, 0.0);  // Инициализация нулями
    MPI_Gatherv(&h_u_curr[1], local_N, MPI_DOUBLE,
                u_final.data(), sendcounts.data(), displs.data(), MPI_DOUBLE,
                0, MPI_COMM_WORLD);
    
    // Проверка корректности данных
    if (rank == 0) {
        bool has_nan = false;
        for (int i = 0; i < N; ++i) {
            if (std::isnan(u_final[i]) || std::isinf(u_final[i])) {
                has_nan = true;
                break;
            }
        }
        if (has_nan) {
            std::cerr << "Warning: NaN or Inf detected in final results!" << std::endl;
        }
    }
    
    // Измерение времени выполнения
    double elapsed_time = std::chrono::duration<double>(end_time - start_time).count();
    double max_time, min_time, avg_time;
    MPI_Reduce(&elapsed_time, &max_time, 1, MPI_DOUBLE, MPI_MAX, 0, MPI_COMM_WORLD);
    MPI_Reduce(&elapsed_time, &min_time, 1, MPI_DOUBLE, MPI_MIN, 0, MPI_COMM_WORLD);
    MPI_Reduce(&elapsed_time, &avg_time, 1, MPI_DOUBLE, MPI_SUM, 0, MPI_COMM_WORLD);
    if (rank == 0) {
        avg_time /= size;
    }
    
    // Вывод результатов
    if (rank == 0) {
        std::cout << std::fixed << std::setprecision(6);
        std::cout << "\nРезультаты производительности:" << std::endl;
        std::cout << "  Время выполнения (мин): " << min_time << " сек" << std::endl;
        std::cout << "  Время выполнения (макс): " << max_time << " сек" << std::endl;
        std::cout << "  Время выполнения (средн): " << avg_time << " сек" << std::endl;
        
        std::cout << "\nПримеры значений температуры:" << std::endl;
        bool all_valid = true;
        for (int i = 0; i <= 10; i += 2) {
            int idx = i * N / 10;
            if (idx >= N) idx = N - 1;
            double value = u_final[idx];
            if (std::isnan(value) || std::isinf(value)) {
                all_valid = false;
                std::cout << "  u(" << (idx * h) << ") = NaN/Inf (invalid)" << std::endl;
            } else {
                std::cout << "  u(" << (idx * h) << ") = " << value << std::endl;
            }
        }
        if (!all_valid) {
            std::cerr << "Warning: Some values are NaN/Inf. This may indicate a problem with GPU computation or data synchronization." << std::endl;
        }
    }
    
    // Освобождение памяти
    cudaFree(d_u_curr);
    cudaFree(d_u_next);
    if (h_boundary_left) cudaFreeHost(h_boundary_left);
    if (h_boundary_right) cudaFreeHost(h_boundary_right);
    if (h_boundary_send_right) cudaFreeHost(h_boundary_send_right);
    
    MPI_Finalize();
    return 0;
}

