#!/usr/bin/env python3
"""
Анализ результатов профилирования OpenMP Matrix Multiplication
Генерирует графики производительности и масштабируемости
"""

import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import os

def load_results(filename='omp_profiling_results.csv'):
    """Загружает результаты профилирования из CSV файла"""
    if not os.path.exists(filename):
        print(f"Файл {filename} не найден. Запустите сначала task4.")
        return None
    
    df = pd.read_csv(filename)
    return df

def plot_performance_by_size(df):
    """График производительности по размерам матриц"""
    plt.figure(figsize=(12, 8))
    
    # Группируем по размеру матрицы и методу
    for method in df['function_name'].unique():
        method_data = df[df['function_name'] == method]
        sizes = method_data['matrix_size'].unique()
        
        # Берем среднее по всем потокам для каждого размера
        avg_gflops = []
        for size in sizes:
            size_data = method_data[method_data['matrix_size'] == size]
            avg_gflops.append(size_data['gflops'].mean())
        
        plt.plot(sizes, avg_gflops, marker='o', label=method, linewidth=2)
    
    plt.xlabel('Размер матрицы (N×N)')
    plt.ylabel('Производительность (GFLOPS)')
    plt.title('Производительность алгоритмов умножения матриц')
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig('performance_by_size.png', dpi=300, bbox_inches='tight')
    plt.show()

def plot_scalability(df):
    """График масштабируемости по количеству потоков"""
    plt.figure(figsize=(12, 8))
    
    # Группируем по методу и размеру матрицы
    for method in df['function_name'].unique():
        if method == 'Sequential':
            continue  # Пропускаем последовательную версию
            
        method_data = df[df['function_name'] == method]
        
        for size in method_data['matrix_size'].unique():
            size_data = method_data[method_data['matrix_size'] == size]
            
            # Рассчитываем speedup относительно последовательной версии
            seq_data = df[(df['function_name'] == 'Sequential') & 
                         (df['matrix_size'] == size)]
            
            if len(seq_data) > 0:
                seq_time = seq_data['execution_time_ms'].iloc[0]
                speedup = seq_time / size_data['execution_time_ms']
                
                plt.plot(size_data['num_threads'], speedup, 
                        marker='o', label=f'{method} ({size}×{size})', 
                        linewidth=2)
    
    plt.xlabel('Количество потоков')
    plt.ylabel('Ускорение (Speedup)')
    plt.title('Масштабируемость OpenMP алгоритмов')
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.axhline(y=1, color='black', linestyle='--', alpha=0.5)
    plt.tight_layout()
    plt.savefig('scalability_analysis.png', dpi=300, bbox_inches='tight')
    plt.show()

def plot_execution_time(df):
    """График времени выполнения"""
    plt.figure(figsize=(12, 8))
    
    # Создаем subplot для каждого размера матрицы
    sizes = sorted(df['matrix_size'].unique())
    n_sizes = len(sizes)
    
    fig, axes = plt.subplots(1, n_sizes, figsize=(5*n_sizes, 6))
    if n_sizes == 1:
        axes = [axes]
    
    for i, size in enumerate(sizes):
        size_data = df[df['matrix_size'] == size]
        
        for method in size_data['function_name'].unique():
            method_data = size_data[size_data['function_name'] == method]
            axes[i].plot(method_data['num_threads'], method_data['execution_time_ms'], 
                        marker='o', label=method, linewidth=2)
        
        axes[i].set_xlabel('Количество потоков')
        axes[i].set_ylabel('Время выполнения (мс)')
        axes[i].set_title(f'Матрица {size}×{size}')
        axes[i].legend()
        axes[i].grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig('execution_time_analysis.png', dpi=300, bbox_inches='tight')
    plt.show()

def generate_summary_report(df):
    """Генерирует текстовый отчет с анализом"""
    print("=" * 60)
    print("ОТЧЕТ АНАЛИЗА ПРОИЗВОДИТЕЛЬНОСТИ")
    print("=" * 60)
    
    # Общая статистика
    print(f"\nОбщее количество тестов: {len(df)}")
    print(f"Размеры матриц: {sorted(df['matrix_size'].unique())}")
    print(f"Количество потоков: {sorted(df['num_threads'].unique())}")
    print(f"Методы: {', '.join(df['function_name'].unique())}")
    
    # Лучшая производительность
    best_performance = df.loc[df['gflops'].idxmax()]
    print(f"\nЛучшая производительность:")
    print(f"  Метод: {best_performance['function_name']}")
    print(f"  Размер: {best_performance['matrix_size']}×{best_performance['matrix_size']}")
    print(f"  Потоки: {best_performance['num_threads']}")
    print(f"  GFLOPS: {best_performance['gflops']:.2f}")
    
    # Анализ масштабируемости
    print(f"\nАнализ масштабируемости:")
    for method in df['function_name'].unique():
        if method == 'Sequential':
            continue
            
        method_data = df[df['function_name'] == method]
        max_threads = method_data['num_threads'].max()
        min_threads = method_data['num_threads'].min()
        
        # Берем данные для максимального размера матрицы
        max_size = method_data['matrix_size'].max()
        size_data = method_data[method_data['matrix_size'] == max_size]
        
        if len(size_data) > 1:
            max_gflops = size_data['gflops'].max()
            min_gflops = size_data['gflops'].min()
            improvement = (max_gflops - min_gflops) / min_gflops * 100
            
            print(f"  {method}: улучшение на {improvement:.1f}% при увеличении потоков с {min_threads} до {max_threads}")

def main():
    """Основная функция анализа"""
    print("Анализ результатов профилирования OpenMP Matrix Multiplication")
    print("=" * 60)
    
    # Загружаем данные
    df = load_results()
    if df is None:
        return
    
    # Устанавливаем стиль графиков
    plt.style.use('seaborn-v0_8')
    sns.set_palette("husl")
    
    # Генерируем графики
    print("Генерация графиков...")
    plot_performance_by_size(df)
    plot_scalability(df)
    plot_execution_time(df)
    
    # Генерируем отчет
    generate_summary_report(df)
    
    print(f"\nАнализ завершен! Графики сохранены:")
    print("  - performance_by_size.png")
    print("  - scalability_analysis.png") 
    print("  - execution_time_analysis.png")

if __name__ == "__main__":
    main()
