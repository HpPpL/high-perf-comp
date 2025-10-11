#!/usr/bin/env python3
"""
Скрипт для анализа результатов масштабируемости DGEMM
Создает графики времени выполнения и GFLOPS vs количество процессов
"""

import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns

plt.style.use('seaborn-v0_8')
sns.set_palette("husl")

def load_results(filename='scalability_results.csv'):
    """Загружает результаты из CSV файла"""
    try:
        df = pd.read_csv(filename)
        print(f"Загружено {len(df)} записей из {filename}")
        return df
    except FileNotFoundError:
        print(f"Файл {filename} не найден!")
        return None

def create_execution_time_plot(df):
    """Создает график времени выполнения vs количество процессов"""
    plt.figure(figsize=(12, 8))
    
    matrix_sizes = sorted(df['N'].unique())
    colors = ['#1f77b4', '#ff7f0e', '#2ca02c']
    
    for i, N in enumerate(matrix_sizes):
        data = df[df['N'] == N]
        plt.plot(data['threads'], data['time_ms'], 
                marker='o', linewidth=2, markersize=8,
                label=f'N = {N}', color=colors[i])
    
    plt.xlabel('Количество процессов (P)', fontsize=12)
    plt.ylabel('Время выполнения (мс)', fontsize=12)
    plt.title('Анализ сильной масштабируемости: Время выполнения vs Количество процессов', 
              fontsize=14, fontweight='bold')
    plt.legend(fontsize=11)
    plt.grid(True, alpha=0.3)
    plt.xscale('log', base=2)
    plt.yscale('log')
    
    # Добавляем аннотации
    plt.annotate('Идеальная масштабируемость', 
                xy=(2, 1000), xytext=(4, 2000),
                arrowprops=dict(arrowstyle='->', color='red', alpha=0.7),
                fontsize=10, color='red')
    
    plt.tight_layout()
    plt.savefig('execution_time_scalability.png', dpi=300, bbox_inches='tight')
    plt.show()

def create_gflops_plot(df):
    """Создает график GFLOPS vs количество процессов"""
    plt.figure(figsize=(12, 8))
    
    matrix_sizes = sorted(df['N'].unique())
    colors = ['#1f77b4', '#ff7f0e', '#2ca02c']
    
    for i, N in enumerate(matrix_sizes):
        data = df[df['N'] == N]
        plt.plot(data['threads'], data['gflops'], 
                marker='s', linewidth=2, markersize=8,
                label=f'N = {N}', color=colors[i])
    
    plt.xlabel('Количество процессов (P)', fontsize=12)
    plt.ylabel('Производительность (GFLOPS)', fontsize=12)
    plt.title('Анализ сильной масштабируемости: GFLOPS vs Количество процессов', 
              fontsize=14, fontweight='bold')
    plt.legend(fontsize=11)
    plt.grid(True, alpha=0.3)
    plt.xscale('log', base=2)
    
    # Добавляем аннотации
    plt.annotate('Линейное ускорение', 
                xy=(8, 50), xytext=(12, 80),
                arrowprops=dict(arrowstyle='->', color='green', alpha=0.7),
                fontsize=10, color='green')
    
    plt.tight_layout()
    plt.savefig('gflops_scalability.png', dpi=300, bbox_inches='tight')
    plt.show()

def calculate_efficiency(df):
    """Вычисляет эффективность масштабируемости"""
    print("\n=== Анализ эффективности масштабируемости ===")
    
    matrix_sizes = sorted(df['N'].unique())
    
    for N in matrix_sizes:
        print(f"\nМатрица {N}x{N}:")
        data = df[df['N'] == N].sort_values('threads')
        
        # Базовое время (1 поток)
        base_time = data[data['threads'] == 1]['time_ms'].iloc[0]
        base_gflops = data[data['threads'] == 1]['gflops'].iloc[0]
        
        print(f"{'P':>3} {'Time(ms)':>10} {'GFLOPS':>8} {'Speedup':>8} {'Efficiency':>10}")
        print("-" * 45)
        
        for _, row in data.iterrows():
            speedup = base_time / row['time_ms']
            efficiency = speedup / row['threads'] * 100
            
            print(f"{row['threads']:>3} {row['time_ms']:>10.2f} {row['gflops']:>8.2f} "
                  f"{speedup:>8.2f} {efficiency:>9.1f}%")

def create_combined_plot(df):
    """Создает комбинированный график с двумя подграфиками"""
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 12))
    
    matrix_sizes = sorted(df['N'].unique())
    colors = ['#1f77b4', '#ff7f0e', '#2ca02c']
    
    # График времени выполнения
    for i, N in enumerate(matrix_sizes):
        data = df[df['N'] == N]
        ax1.plot(data['threads'], data['time_ms'], 
                marker='o', linewidth=2, markersize=8,
                label=f'N = {N}', color=colors[i])
    
    ax1.set_xlabel('Количество процессов (P)', fontsize=12)
    ax1.set_ylabel('Время выполнения (мс)', fontsize=12)
    ax1.set_title('Время выполнения vs Количество процессов', fontsize=13, fontweight='bold')
    ax1.legend(fontsize=11)
    ax1.grid(True, alpha=0.3)
    ax1.set_xscale('log', base=2)
    ax1.set_yscale('log')
    
    # График GFLOPS
    for i, N in enumerate(matrix_sizes):
        data = df[df['N'] == N]
        ax2.plot(data['threads'], data['gflops'], 
                marker='s', linewidth=2, markersize=8,
                label=f'N = {N}', color=colors[i])
    
    ax2.set_xlabel('Количество процессов (P)', fontsize=12)
    ax2.set_ylabel('Производительность (GFLOPS)', fontsize=12)
    ax2.set_title('GFLOPS vs Количество процессов', fontsize=13, fontweight='bold')
    ax2.legend(fontsize=11)
    ax2.grid(True, alpha=0.3)
    ax2.set_xscale('log', base=2)
    
    plt.suptitle('Анализ сильной масштабируемости DGEMM на суперкомпьютере Харизма', 
                 fontsize=16, fontweight='bold')
    plt.tight_layout()
    plt.savefig('combined_scalability_analysis.png', dpi=300, bbox_inches='tight')
    plt.show()

def main():
    """Основная функция"""
    print("=== Анализ масштабируемости DGEMM ===")
    
    # Загружаем результаты
    df = load_results()
    if df is None:
        return
    
    # Показываем общую статистику
    print(f"\nОбщая статистика:")
    print(f"Размеры матриц: {sorted(df['N'].unique())}")
    print(f"Количество процессов: {sorted(df['threads'].unique())}")
    print(f"Общее количество измерений: {len(df)}")
    
    # Создаем графики
    print("\nСоздание графиков...")
    create_execution_time_plot(df)
    create_gflops_plot(df)
    create_combined_plot(df)
    
    # Анализ эффективности
    calculate_efficiency(df)
    
    print("\n=== Анализ завершен ===")
    print("Созданные файлы:")
    print("- execution_time_scalability.png")
    print("- gflops_scalability.png") 
    print("- combined_scalability_analysis.png")

if __name__ == "__main__":
    main()
