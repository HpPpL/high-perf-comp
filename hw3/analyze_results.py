#!/usr/bin/env python3
"""
Скрипт для анализа результатов решения уравнения теплопроводности с MPI
Создает графики масштабируемости и сравнения методов
"""

import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
import os
import sys

# Настройка стиля
plt.rcParams['font.size'] = 11
plt.rcParams['figure.figsize'] = (14, 8)
plt.rcParams['figure.dpi'] = 100
sns.set_style("whitegrid")
sns.set_palette("husl")

def load_results(task_name, filename):
    """Загружает результаты из CSV файла"""
    filepath = os.path.join(task_name, filename)
    if os.path.exists(filepath):
        try:
            df = pd.read_csv(filepath)
            print(f"  Загружено {len(df)} записей из {filepath}")
            return df
        except Exception as e:
            print(f"  Ошибка при загрузке {filepath}: {e}")
            return None
    return None

def plot_task1_correctness():
    """График сравнения численного и точного решений для Task 1"""
    print("Создание графика корректности (Task 1)...")
    
    # Загружаем результаты из файла task1_results.txt
    filepath = 'task1/task1_results.txt'
    if not os.path.exists(filepath):
        print(f"  Файл {filepath} не найден, пропускаем Task 1")
        return
    
    try:
        data = pd.read_csv(filepath, sep=' ', header=None, names=['x', 'u_numerical'])
        
        # Вычисляем точное решение
        k, u0, l, T = 1.0, 1.0, 1.0, 0.1
        pi = np.pi
        
        def exact_solution(x, t, max_terms=100):
            sum_val = 0.0
            for m in range(max_terms):
                n = 2 * m + 1
                coeff = 1.0 / n
                exp_term = np.exp(-k * pi * pi * n * n * t / (l * l))
                sin_term = np.sin(pi * n * x / l)
                sum_val += coeff * exp_term * sin_term
            return (4.0 * u0 / pi) * sum_val
        
        data['u_exact'] = data['x'].apply(lambda x: exact_solution(x, T))
        data['error'] = np.abs(data['u_numerical'] - data['u_exact'])
        
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 6))
        
        # График сравнения решений
        ax1.plot(data['x'], data['u_numerical'], 'o-', label='Численное решение', markersize=6)
        ax1.plot(data['x'], data['u_exact'], 's-', label='Точное решение', markersize=6)
        ax1.set_xlabel('x', fontsize=12)
        ax1.set_ylabel('u(x, T)', fontsize=12)
        ax1.set_title('Сравнение численного и точного решений (T = 0.1)', fontsize=14, fontweight='bold')
        ax1.legend(fontsize=11)
        ax1.grid(True, alpha=0.3)
        
        # График ошибок
        ax2.plot(data['x'], data['error'], 'r-o', markersize=6)
        ax2.set_xlabel('x', fontsize=12)
        ax2.set_ylabel('Абсолютная ошибка', fontsize=12)
        ax2.set_title('Ошибка численного решения', fontsize=14, fontweight='bold')
        ax2.grid(True, alpha=0.3)
        ax2.set_yscale('log')
        
        plt.tight_layout()
        plt.savefig('task1_correctness.png', dpi=150, bbox_inches='tight')
        print("  Сохранено: task1_correctness.png")
        plt.close()
        
    except Exception as e:
        print(f"  Ошибка при обработке Task 1: {e}")

def plot_task2_scalability():
    """График масштабируемости для Task 2"""
    print("Создание графика масштабируемости (Task 2)...")
    
    df = load_results('task2', 'scalability_results.csv')
    if df is None or len(df) == 0:
        print("  Нет данных для Task 2, пропускаем")
        return
    
    # Создаем графики для каждого размера сетки
    N_values = sorted(df['N'].unique())
    fig, axes = plt.subplots(1, len(N_values), figsize=(6*len(N_values), 6))
    
    if len(N_values) == 1:
        axes = [axes]
    
    for idx, N in enumerate(N_values):
        data = df[df['N'] == N].sort_values('num_procs')
        
        ax = axes[idx]
        ax.plot(data['num_procs'], data['avg_time'], 'o-', linewidth=2, markersize=8, label='Среднее время')
        ax.fill_between(data['num_procs'], data['min_time'], data['max_time'], 
                        alpha=0.3, label='Диапазон')
        
        ax.set_xlabel('Количество процессов', fontsize=12)
        ax.set_ylabel('Время выполнения (сек)', fontsize=12)
        ax.set_title(f'Масштабируемость (N = {N})', fontsize=14, fontweight='bold')
        ax.legend(fontsize=11)
        ax.grid(True, alpha=0.3)
        ax.set_xscale('log', base=2)
        ax.set_yscale('log')
    
    plt.tight_layout()
    plt.savefig('task2_scalability.png', dpi=150, bbox_inches='tight')
    print("  Сохранено: task2_scalability.png")
    plt.close()

def plot_task3_comparison():
    """Сравнение синхронной и асинхронной версий"""
    print("Создание графика сравнения (Task 3)...")
    
    df_sync = load_results('task2', 'scalability_results.csv')
    df_async = load_results('task3', 'scalability_results_async.csv')
    
    if df_sync is None or df_async is None:
        print("  Нет данных для сравнения, пропускаем")
        return
    
    # Объединяем данные
    df_sync['method'] = 'Синхронная'
    df_async['method'] = 'Асинхронная'
    df_combined = pd.concat([df_sync, df_async], ignore_index=True)
    
    # График сравнения
    N_values = sorted(df_combined['N'].unique())
    fig, axes = plt.subplots(1, len(N_values), figsize=(6*len(N_values), 6))
    
    if len(N_values) == 1:
        axes = [axes]
    
    for idx, N in enumerate(N_values):
        data = df_combined[df_combined['N'] == N].sort_values(['method', 'num_procs'])
        
        ax = axes[idx]
        for method in ['Синхронная', 'Асинхронная']:
            method_data = data[data['method'] == method].sort_values('num_procs')
            ax.plot(method_data['num_procs'], method_data['avg_time'], 
                   'o-', linewidth=2, markersize=8, label=method)
        
        ax.set_xlabel('Количество процессов', fontsize=12)
        ax.set_ylabel('Время выполнения (сек)', fontsize=12)
        ax.set_title(f'Сравнение методов (N = {N})', fontsize=14, fontweight='bold')
        ax.legend(fontsize=11)
        ax.grid(True, alpha=0.3)
        ax.set_xscale('log', base=2)
        ax.set_yscale('log')
    
    plt.tight_layout()
    plt.savefig('task3_comparison.png', dpi=150, bbox_inches='tight')
    print("  Сохранено: task3_comparison.png")
    plt.close()

def plot_task4_performance():
    """График производительности с коллективными операциями"""
    print("Создание графика производительности (Task 4)...")
    
    df = load_results('task4', 'scalability_results_collective.csv')
    if df is None or len(df) == 0:
        print("  Нет данных для Task 4, пропускаем")
        return
    
    # Аналогично Task 2
    N_values = sorted(df['N'].unique())
    fig, axes = plt.subplots(1, len(N_values), figsize=(6*len(N_values), 6))
    
    if len(N_values) == 1:
        axes = [axes]
    
    for idx, N in enumerate(N_values):
        data = df[df['N'] == N].sort_values('num_procs')
        
        ax = axes[idx]
        ax.plot(data['num_procs'], data['avg_time'], 'o-', linewidth=2, markersize=8, 
               label='Среднее время', color='green')
        ax.fill_between(data['num_procs'], data['min_time'], data['max_time'], 
                        alpha=0.3, label='Диапазон')
        
        ax.set_xlabel('Количество процессов', fontsize=12)
        ax.set_ylabel('Время выполнения (сек)', fontsize=12)
        ax.set_title(f'Производительность с коллективными операциями (N = {N})', 
                    fontsize=14, fontweight='bold')
        ax.legend(fontsize=11)
        ax.grid(True, alpha=0.3)
        ax.set_xscale('log', base=2)
        ax.set_yscale('log')
    
    plt.tight_layout()
    plt.savefig('task4_performance.png', dpi=150, bbox_inches='tight')
    print("  Сохранено: task4_performance.png")
    plt.close()

def plot_task6_2d_results():
    """Визуализация 2D решения"""
    print("Создание визуализации 2D решения (Task 6)...")
    
    filepath = 'task6/task6_results.txt'
    if not os.path.exists(filepath):
        print(f"  Файл {filepath} не найден, пропускаем Task 6")
        return
    
    try:
        data = pd.read_csv(filepath, sep=' ', header=None, names=['x', 'y', 'u'])
        
        # Определяем размер сетки
        x_unique = sorted(data['x'].unique())
        y_unique = sorted(data['y'].unique())
        N = len(x_unique)
        
        # Преобразуем в матрицу для визуализации
        u_matrix = data.pivot(index='y', columns='x', values='u').values
        
        fig, ax = plt.subplots(figsize=(10, 8))
        
        # Контурный график
        X, Y = np.meshgrid(x_unique, y_unique)
        contour = ax.contourf(X, Y, u_matrix, levels=20, cmap='hot')
        ax.contour(X, Y, u_matrix, levels=20, colors='black', alpha=0.3, linewidths=0.5)
        
        ax.set_xlabel('x', fontsize=12)
        ax.set_ylabel('y', fontsize=12)
        ax.set_title('Распределение температуры в момент времени T = 0.01', 
                    fontsize=14, fontweight='bold')
        
        cbar = plt.colorbar(contour, ax=ax)
        cbar.set_label('Температура u(x,y)', fontsize=12)
        
        plt.tight_layout()
        plt.savefig('task6_2d_results.png', dpi=150, bbox_inches='tight')
        print("  Сохранено: task6_2d_results.png")
        plt.close()
        
    except Exception as e:
        print(f"  Ошибка при обработке Task 6: {e}")

def main():
    """Основная функция"""
    print("=" * 60)
    print("Анализ результатов решения уравнения теплопроводности (MPI)")
    print("=" * 60)
    print()
    
    # Создаем папку для графиков
    os.makedirs('plots', exist_ok=True)
    
    # Генерируем графики для каждой задачи
    plot_task1_correctness()
    plot_task2_scalability()
    plot_task3_comparison()
    plot_task4_performance()
    plot_task6_2d_results()
    
    print()
    print("=" * 60)
    print("Анализ завершен!")
    print("=" * 60)

if __name__ == '__main__':
    main()

