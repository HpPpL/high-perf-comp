#include <mpi.h>
#include <iostream>
#include <iomanip>
#include <cmath>
#include <vector>
#include <fstream>
#include <chrono>

int main(int argc, char* argv[]) {
    MPI_Init(&argc, &argv);
    
    int rank, size;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &size);
    
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
        std::cout << "=== Task 3: Асинхронная передача данных ===" << std::endl;
        std::cout << "Параметры:" << std::endl;
        std::cout << "  N = " << N << " точек" << std::endl;
        std::cout << "  h = " << h << std::endl;
        std::cout << "  τ = " << tau << std::endl;
        std::cout << "  T = " << T << std::endl;
        std::cout << "  Количество шагов по времени: " << num_time_steps << std::endl;
        std::cout << "  Количество процессов: " << size << std::endl;
    }
    
    // Распределение точек между процессами
    int points_per_proc = N / size;
    int remainder = N % size;
    
    int local_start = rank * points_per_proc + std::min(rank, remainder);
    int local_end = local_start + points_per_proc + (rank < remainder ? 1 : 0);
    int local_N = local_end - local_start;
    
    // Выделение памяти
    std::vector<double> u_curr(local_N + 2);
    std::vector<double> u_next(local_N + 2);
    
    // Инициализация начального распределения
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
                 &u_curr[1], local_N, MPI_DOUBLE, 0, MPI_COMM_WORLD);
    
    // Граничные условия
    if (rank == 0) {
        u_curr[0] = 0.0;
    }
    if (rank == size - 1) {
        u_curr[local_N + 1] = 0.0;
    }
    
    // Запросы для асинхронных операций
    MPI_Request send_left_req, send_right_req;
    MPI_Request recv_left_req, recv_right_req;
    
    MPI_Barrier(MPI_COMM_WORLD);
    auto start_time = std::chrono::high_resolution_clock::now();
    
    double coeff = k * tau / (h * h);
    
    // Основной цикл по времени
    for (int n = 0; n < num_time_steps; ++n) {
        // Инициализация асинхронных операций обмена граничными значениями
        if (rank > 0) {
            // Отправка левой границы предыдущему процессу
            MPI_Isend(&u_curr[1], 1, MPI_DOUBLE, rank - 1, 0, MPI_COMM_WORLD, &send_left_req);
            // Получение правой границы от предыдущего процесса
            MPI_Irecv(&u_curr[0], 1, MPI_DOUBLE, rank - 1, 0, MPI_COMM_WORLD, &recv_left_req);
        }
        
        if (rank < size - 1) {
            // Получение левой границы от следующего процесса
            MPI_Irecv(&u_curr[local_N + 1], 1, MPI_DOUBLE, rank + 1, 0, MPI_COMM_WORLD, &recv_right_req);
            // Отправка правой границы следующему процессу
            MPI_Isend(&u_curr[local_N], 1, MPI_DOUBLE, rank + 1, 0, MPI_COMM_WORLD, &send_right_req);
        }
        
        // Ожидание завершения получения граничных значений (необходимо для вычислений)
        if (rank > 0) {
            MPI_Wait(&recv_left_req, MPI_STATUS_IGNORE);
        }
        
        if (rank < size - 1) {
            MPI_Wait(&recv_right_req, MPI_STATUS_IGNORE);
        }
        
        // Теперь можно выполнить все вычисления, так как граничные значения получены
        // Вычисление нового временного слоя
        for (int i = 1; i <= local_N; ++i) {
            u_next[i] = u_curr[i] + coeff * (u_curr[i+1] - 2.0 * u_curr[i] + u_curr[i-1]);
        }
        
        // Ожидание завершения отправки (для синхронизации, но вычисления уже выполнены)
        if (rank > 0) {
            MPI_Wait(&send_left_req, MPI_STATUS_IGNORE);
        }
        
        if (rank < size - 1) {
            MPI_Wait(&send_right_req, MPI_STATUS_IGNORE);
        }
        
        // Граничные условия
        if (rank == 0) {
            u_next[0] = 0.0;
        }
        if (rank == size - 1) {
            u_next[local_N + 1] = 0.0;
        }
        
        std::swap(u_curr, u_next);
    }
    
    MPI_Barrier(MPI_COMM_WORLD);
    auto end_time = std::chrono::high_resolution_clock::now();
    
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
        
        // Сохранение результатов в CSV
        std::ofstream csv("scalability_results_async.csv", std::ios::app);
        csv << std::fixed << std::setprecision(6);
        csv << size << "," << N << "," << min_time << "," << max_time << "," << avg_time << std::endl;
        csv.close();
        
        std::cout << "Результаты добавлены в scalability_results_async.csv" << std::endl;
    }
    
    MPI_Finalize();
    return 0;
}

