#include <mkl.h>
#include <iostream>
#include <iomanip>
#include <cmath>
#include <vector>
#include <fstream>
#include <chrono>

int main(int argc, char* argv[]) {
    // Параметры задачи
    const double k = 1.0;           // коэффициент температуропроводности
    double h;                        // шаг по пространству
    const double l = 1.0;           // длина области
    const double u0 = 1.0;          // начальная температура
    const double T = 0.01;          // конечное время
    
    // Размер сетки
    int N = 100;
    if (argc > 1) {
        N = std::atoi(argv[1]);
    }
    
    h = l / (N - 1);
    
    // Вычисление шага по времени из условия устойчивости для 2D
    // Для 2D: k*τ/h² < 0.25 (более строгое условие)
    double tau = 0.2 * h * h / k;
    int num_time_steps = static_cast<int>(T / tau);
    if (num_time_steps == 0) num_time_steps = 1;
    
    std::cout << "=== Task 6: 2D Теплопроводность с MKL ===" << std::endl;
    std::cout << "Параметры:" << std::endl;
    std::cout << "  N = " << N << "x" << N << " точек" << std::endl;
    std::cout << "  h = " << h << std::endl;
    std::cout << "  τ = " << tau << std::endl;
    std::cout << "  T = " << T << std::endl;
    std::cout << "  Количество шагов по времени: " << num_time_steps << std::endl;
    
    // Выделение памяти для 2D сетки
    int total_points = N * N;
    std::vector<double> u_curr(total_points);
    std::vector<double> u_next(total_points);
    
    // Инициализация начального распределения температуры
    // Начальное условие: u(x,y,0) = u0 для (x,y) внутри области [0,1]x[0,1]
    for (int i = 0; i < N; ++i) {
        for (int j = 0; j < N; ++j) {
            double x = i * h;
            double y = j * h;
            int idx = i * N + j;
            
            if (x >= 0.0 && x <= l && y >= 0.0 && y <= l) {
                u_curr[idx] = u0;
            } else {
                u_curr[idx] = 0.0;
            }
        }
    }
    
    // Граничные условия Дирихле: u = 0 на границе
    for (int i = 0; i < N; ++i) {
        u_curr[i] = 0.0;                    // нижняя граница (y=0)
        u_curr[(N-1)*N + i] = 0.0;          // верхняя граница (y=1)
        u_curr[i*N] = 0.0;                   // левая граница (x=0)
        u_curr[i*N + (N-1)] = 0.0;          // правая граница (x=1)
    }
    
    double coeff = k * tau / (h * h);
    
    auto start_time = std::chrono::high_resolution_clock::now();
    
    // Основной цикл по времени
    for (int n = 0; n < num_time_steps; ++n) {
        // Решение 2D уравнения теплопроводности с использованием MKL
        // Явная схема: u^(n+1) = u^n + (kτ/h²) * (Δu^n)
        // где Δu = u_(i+1,j) + u_(i-1,j) + u_(i,j+1) + u_(i,j-1) - 4*u_(i,j)
        
        // Используем MKL для эффективных операций с матрицами
        // Копируем текущее состояние
        cblas_dcopy(total_points, u_curr.data(), 1, u_next.data(), 1);
        
        // Вычисление лапласиана и обновление
        for (int i = 1; i < N - 1; ++i) {
            for (int j = 1; j < N - 1; ++j) {
                int idx = i * N + j;
                double laplacian = u_curr[(i+1)*N + j] + u_curr[(i-1)*N + j] +
                                   u_curr[i*N + (j+1)] + u_curr[i*N + (j-1)] -
                                   4.0 * u_curr[idx];
                u_next[idx] += coeff * laplacian;
            }
        }
        
        // Применение граничных условий
        for (int i = 0; i < N; ++i) {
            u_next[i] = 0.0;                    // нижняя граница
            u_next[(N-1)*N + i] = 0.0;          // верхняя граница
            u_next[i*N] = 0.0;                   // левая граница
            u_next[i*N + (N-1)] = 0.0;          // правая граница
        }
        
        // Обновление для следующей итерации
        std::swap(u_curr, u_next);
    }
    
    auto end_time = std::chrono::high_resolution_clock::now();
    double elapsed_time = std::chrono::duration<double>(end_time - start_time).count();
    
    std::cout << std::fixed << std::setprecision(6);
    std::cout << "\nРезультаты производительности:" << std::endl;
    std::cout << "  Время выполнения: " << elapsed_time << " сек" << std::endl;
    
    // Вывод некоторых значений результата
    std::cout << "\nПримеры значений температуры в момент времени T = " << T << ":" << std::endl;
    std::cout << std::setw(8) << "x" << std::setw(8) << "y" << std::setw(15) << "u(x,y)" << std::endl;
    std::cout << std::string(31, '-') << std::endl;
    
    for (int i = 0; i <= 4; ++i) {
        for (int j = 0; j <= 4; ++j) {
            int idx_i = i * (N - 1) / 4;
            int idx_j = j * (N - 1) / 4;
            int idx = idx_i * N + idx_j;
            double x = idx_i * h;
            double y = idx_j * h;
            std::cout << std::setw(8) << x << std::setw(8) << y 
                      << std::setw(15) << u_curr[idx] << std::endl;
        }
    }
    
    // Сохранение результатов в файл
    std::ofstream out("task6_results.txt");
    out << std::fixed << std::setprecision(10);
    for (int i = 0; i < N; ++i) {
        for (int j = 0; j < N; ++j) {
            double x = i * h;
            double y = j * h;
            int idx = i * N + j;
            out << x << " " << y << " " << u_curr[idx] << std::endl;
        }
    }
    out.close();
    std::cout << "\nРезультаты сохранены в task6_results.txt" << std::endl;
    
    // Сохранение результатов в CSV для анализа
    std::ofstream csv("task6_performance.csv", std::ios::app);
    csv << std::fixed << std::setprecision(6);
    csv << N << "," << elapsed_time << std::endl;
    csv.close();
    
    return 0;
}

