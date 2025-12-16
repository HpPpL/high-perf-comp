#include <mpi.h>
#include <iostream>
#include <iomanip>
#include <cmath>
#include <vector>
#include <fstream>
#include <algorithm>

// Точное решение уравнения теплопроводности
// u(x,t) = (4u₀/π) * Σ[m=0 to ∞] (1/(2m+1)) * exp(-kπ²(2m+1)²t/l²) * sin(π(2m+1)x/l)
double exact_solution(double x, double t, double u0, double k, double l, int max_terms = 100) {
    double sum = 0.0;
    double pi = M_PI;
    
    for (int m = 0; m < max_terms; ++m) {
        int n = 2 * m + 1;
        double coeff = 1.0 / n;
        double exp_term = exp(-k * pi * pi * n * n * t / (l * l));
        double sin_term = sin(pi * n * x / l);
        sum += coeff * exp_term * sin_term;
    }
    
    return (4.0 * u0 / pi) * sum;
}

int main(int argc, char* argv[]) {
    MPI_Init(&argc, &argv);
    
    int rank, size;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &size);
    
    // Параметры задачи
    const double k = 1.0;           // коэффициент температуропроводности
    const double h = 0.02;          // шаг по пространству
    const double tau = 0.0002;      // шаг по времени
    const double l = 1.0;           // длина стержня
    const double u0 = 1.0;          // начальная температура
    const double T = 0.1;           // конечное время
    
    // Проверка условия устойчивости
    double stability = k * tau / (h * h);
    if (rank == 0) {
        std::cout << "=== Task 1: Проверка корректности алгоритма ===" << std::endl;
        std::cout << "Параметры:" << std::endl;
        std::cout << "  k = " << k << std::endl;
        std::cout << "  h = " << h << std::endl;
        std::cout << "  τ = " << tau << std::endl;
        std::cout << "  T = " << T << std::endl;
        std::cout << "  Условие устойчивости: kτ/h² = " << stability << std::endl;
        if (stability >= 1.0) {
            std::cerr << "ОШИБКА: Условие устойчивости не выполнено!" << std::endl;
            MPI_Abort(MPI_COMM_WORLD, 1);
        }
        std::cout << "  Количество процессов: " << size << std::endl;
    }
    
    // Количество точек по пространству
    int N = static_cast<int>(l / h) + 1;
    int num_time_steps = static_cast<int>(T / tau);
    
    // Распределение точек между процессами
    int points_per_proc = N / size;
    int remainder = N % size;
    
    int local_start = rank * points_per_proc + std::min(rank, remainder);
    int local_end = local_start + points_per_proc + (rank < remainder ? 1 : 0);
    int local_N = local_end - local_start;
    
    // Выделение памяти для текущего и следующего временного слоя
    std::vector<double> u_curr(local_N + 2);  // +2 для граничных точек
    std::vector<double> u_next(local_N + 2);
    
    // Инициализация начального распределения температуры (только на root процессе)
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
        // Граничные условия
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
    
    // Установка граничных условий для локальных массивов
    if (rank == 0) {
        u_curr[0] = 0.0;  // левая граница
    }
    if (rank == size - 1) {
        u_curr[local_N + 1] = 0.0;  // правая граница
    }
    
    // Основной цикл по времени
    for (int n = 0; n < num_time_steps; ++n) {
        // Обмен граничными значениями между процессами
        if (rank > 0) {
            // Отправка левой границы предыдущему процессу
            MPI_Send(&u_curr[1], 1, MPI_DOUBLE, rank - 1, 0, MPI_COMM_WORLD);
            // Получение правой границы от предыдущего процесса
            MPI_Recv(&u_curr[0], 1, MPI_DOUBLE, rank - 1, 0, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
        }
        
        if (rank < size - 1) {
            // Получение левой границы от следующего процесса
            MPI_Recv(&u_curr[local_N + 1], 1, MPI_DOUBLE, rank + 1, 0, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
            // Отправка правой границы следующему процессу
            MPI_Send(&u_curr[local_N], 1, MPI_DOUBLE, rank + 1, 0, MPI_COMM_WORLD);
        }
        
        // Вычисление нового временного слоя
        double coeff = k * tau / (h * h);
        for (int i = 1; i <= local_N; ++i) {
            u_next[i] = u_curr[i] + coeff * (u_curr[i+1] - 2.0 * u_curr[i] + u_curr[i-1]);
        }
        
        // Применение граничных условий
        if (rank == 0) {
            u_next[0] = 0.0;
        }
        if (rank == size - 1) {
            u_next[local_N + 1] = 0.0;
        }
        
        // Обновление для следующей итерации
        std::swap(u_curr, u_next);
    }
    
    // Сбор результатов на root процессе
    std::vector<double> u_final(N);
    MPI_Gatherv(&u_curr[1], local_N, MPI_DOUBLE,
                u_final.data(), sendcounts.data(), displs.data(), MPI_DOUBLE,
                0, MPI_COMM_WORLD);
    
    // Вывод результатов и сравнение с точным решением
    if (rank == 0) {
        std::cout << "\nРезультаты в момент времени T = " << T << ":" << std::endl;
        std::cout << std::fixed << std::setprecision(6);
        std::cout << std::setw(8) << "x" << std::setw(15) << "Численное" 
                  << std::setw(15) << "Точное" << std::setw(15) << "Ошибка" << std::endl;
        std::cout << std::string(53, '-') << std::endl;
        
        // Вычисление значений в 11 точках: 0, 0.1, 0.2, ..., 1.0
        std::vector<double> errors;
        for (int i = 0; i <= 10; ++i) {
            double x = i * 0.1;
            int idx = static_cast<int>(std::round(x / h));
            if (idx >= N) idx = N - 1;
            
            double numerical = u_final[idx];
            double exact = exact_solution(x, T, u0, k, l);
            double error = std::abs(numerical - exact);
            errors.push_back(error);
            
            std::cout << std::setw(8) << x 
                      << std::setw(15) << numerical
                      << std::setw(15) << exact
                      << std::setw(15) << error << std::endl;
        }
        
        // Статистика ошибок
        double max_error = *std::max_element(errors.begin(), errors.end());
        double avg_error = 0.0;
        for (double e : errors) avg_error += e;
        avg_error /= errors.size();
        
        std::cout << "\nСтатистика ошибок:" << std::endl;
        std::cout << "  Максимальная ошибка: " << max_error << std::endl;
        std::cout << "  Средняя ошибка: " << avg_error << std::endl;
        
        // Сохранение результатов в файл
        std::ofstream out("task1_results.txt");
        out << std::fixed << std::setprecision(10);
        for (int i = 0; i < N; ++i) {
            double x = i * h;
            out << x << " " << u_final[i] << std::endl;
        }
        out.close();
        std::cout << "\nРезультаты сохранены в task1_results.txt" << std::endl;
    }
    
    MPI_Finalize();
    return 0;
}

