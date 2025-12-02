#!/usr/bin/env python3
"""
Скрипт для анализа результатов CUDA матричного умножения
Создает графики времени выполнения и производительности
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

def load_all_results():
    """Загружает результаты из всех CSV файлов"""
    print("Загрузка результатов...")
    results = []
    
    # Task 1: Global memory
    df1 = load_results('task1', 'task1_results.csv')
    if df1 is not None:
        results.append(df1)
    
    # Task 2a: Unified Memory
    df2a = load_results('task2a', 'task2a_results.csv')
    if df2a is not None:
        results.append(df2a)
    
    # Task 2b: All variants (global, unified, streams, shared, shared_opt, cublas)
    df2b = load_results('task2b', 'task2b_results.csv')
    if df2b is not None:
        # Преобразуем формат task2b в совместимый формат
        # Формат task2b: variant,N,param1,param2,time_ms,gflops
        # Преобразуем в: task,N,time_ms,gflops (и дополнительные колонки)
        df2b_processed = df2b.copy()
        df2b_processed['task'] = df2b_processed['variant']
        
        # Для streams: param1 = num_streams, param2 = block_size
        # Для других: param1 = block_size, param2 = 0
        if 'param1' in df2b_processed.columns:
            # Создаем отдельные колонки для удобства
            df2b_processed['num_streams'] = df2b_processed.apply(
                lambda row: int(row['param1']) if row['variant'] == 'streams' else None, axis=1
            )
            df2b_processed['block_size'] = df2b_processed.apply(
                lambda row: int(row['param2']) if row['variant'] == 'streams' else int(row['param1']), axis=1
            )
        
        results.append(df2b_processed)
    
    # Task 2c: Shared Memory
    df2c = load_results('task2c', 'task2c_results.csv')
    if df2c is not None:
        results.append(df2c)
    
    # Task 4: cuBLAS
    df4 = load_results('task4', 'task4_results.csv')
    if df4 is not None:
        results.append(df4)
    
    if results:
        combined = pd.concat(results, ignore_index=True)
        print(f"Всего загружено {len(combined)} записей\n")
        return combined
    return None

def plot_comparison_by_size(df):
    """График сравнения методов по размеру матрицы"""
    print("Создание графика сравнения методов...")
    
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 7))
    
    methods = df['task'].unique()
    sizes = sorted(df['N'].unique())
    
    # График времени выполнения
    for method in methods:
        data = df[df['task'] == method]
        times = []
        for n in sizes:
            matching = data[data['N'] == n]
            if len(matching) > 0:
                # Use mean to average multiple runs for the same N
                times.append(matching['time_ms'].mean())
            else:
                times.append(np.nan)
        ax1.plot(sizes, times, marker='o', linewidth=2.5, markersize=10, 
                label=method, markerfacecolor='white', markeredgewidth=2)
    
    ax1.set_xlabel('Размер матрицы N', fontsize=13, fontweight='bold')
    ax1.set_ylabel('Время выполнения (мс)', fontsize=13, fontweight='bold')
    ax1.set_title('Время выполнения vs Размер матрицы', fontsize=14, fontweight='bold')
    ax1.legend(fontsize=11, loc='best')
    ax1.grid(True, alpha=0.3, linestyle='--')
    ax1.set_xscale('log', base=2)
    ax1.set_yscale('log')
    
    # График производительности
    for method in methods:
        data = df[df['task'] == method]
        gflops = []
        for n in sizes:
            matching = data[data['N'] == n]
            if len(matching) > 0:
                gflops.append(matching['gflops'].iloc[0])
            else:
                gflops.append(np.nan)
        ax2.plot(sizes, gflops, marker='s', linewidth=2.5, markersize=10, 
                label=method, markerfacecolor='white', markeredgewidth=2)
    
    ax2.set_xlabel('Размер матрицы N', fontsize=13, fontweight='bold')
    ax2.set_ylabel('Производительность (GFLOPS)', fontsize=13, fontweight='bold')
    ax2.set_title('Производительность vs Размер матрицы', fontsize=14, fontweight='bold')
    ax2.legend(fontsize=11, loc='best')
    ax2.grid(True, alpha=0.3, linestyle='--')
    ax2.set_xscale('log', base=2)
    
    plt.tight_layout()
    plt.savefig('performance_comparison.png', dpi=300, bbox_inches='tight')
    print("  Сохранен график: performance_comparison.png")
    plt.close()

def plot_streams_analysis():
    """Анализ Task 2b: зависимость от количества streams"""
    print("Создание графика анализа streams...")
    
    df = load_results('task2b', 'task2b_results.csv')
    if df is None or len(df) == 0:
        print("  Файл task2b_results.csv не найден или пуст")
        return
    
    # Фильтруем только streams варианты
    df_streams = df[df['variant'] == 'streams'].copy()
    if len(df_streams) == 0:
        print("  Нет данных streams в task2b_results.csv")
        return
    
    # param1 для streams = num_streams, param2 = block_size
    df_streams['num_streams'] = df_streams['param1'].astype(int)
    df_streams['block_size'] = df_streams['param2'].astype(int)
    
    # Фильтруем по фиксированному block_size=16 для анализа streams
    original_len = len(df_streams)
    df_filtered = df_streams[df_streams['block_size'] == 16]
    if len(df_filtered) > 0:
        df = df_filtered
        print(f"  Используем данные с block_size=16 ({len(df)} из {original_len} записей)")
    else:
        print(f"  Нет данных с block_size=16, используем все данные ({len(df_streams)} записей)")
        df = df_streams
    
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 7))
    
    streams = sorted(df['num_streams'].unique())
    sizes = sorted(df['N'].unique())
    
    # График времени от количества streams
    for size in sizes:
        data = df[df['N'] == size]
        times = []
        for s in streams:
            matching = data[data['num_streams'] == s]
            if len(matching) > 0:
                # Если несколько записей (разные block_size), берем среднее
                times.append(matching['time_ms'].mean())
            else:
                times.append(np.nan)
        ax1.plot(streams, times, marker='o', linewidth=2.5, markersize=10, 
                label=f'N={size}', markerfacecolor='white', markeredgewidth=2)
    
    ax1.set_xlabel('Количество streams', fontsize=13, fontweight='bold')
    ax1.set_ylabel('Время выполнения (мс)', fontsize=13, fontweight='bold')
    ax1.set_title('Время выполнения vs Количество streams', fontsize=14, fontweight='bold')
    ax1.legend(fontsize=11, loc='best')
    ax1.grid(True, alpha=0.3, linestyle='--')
    
    # График производительности от количества streams
    for size in sizes:
        data = df[df['N'] == size]
        gflops = []
        for s in streams:
            matching = data[data['num_streams'] == s]
            if len(matching) > 0:
                # Если несколько записей (разные block_size), берем среднее
                gflops.append(matching['gflops'].mean())
            else:
                gflops.append(np.nan)
        ax2.plot(streams, gflops, marker='s', linewidth=2.5, markersize=10, 
                label=f'N={size}', markerfacecolor='white', markeredgewidth=2)
    
    ax2.set_xlabel('Количество streams', fontsize=13, fontweight='bold')
    ax2.set_ylabel('Производительность (GFLOPS)', fontsize=13, fontweight='bold')
    ax2.set_title('Производительность vs Количество streams', fontsize=14, fontweight='bold')
    ax2.legend(fontsize=11, loc='best')
    ax2.grid(True, alpha=0.3, linestyle='--')
    
    plt.tight_layout()
    plt.savefig('streams_analysis.png', dpi=300, bbox_inches='tight')
    print("  Сохранен график: streams_analysis.png")
    plt.close()

def plot_bank_conflicts():
    """Анализ Task 3: сравнение версий с padding и без"""
    print("Создание графика анализа bank conflicts...")
    
    df = load_results('task3', 'task3_results.csv')
    if df is None or len(df) == 0:
        print("  Файл task3_results.csv не найден или пуст")
        return
    
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 7))
    
    versions = df['version'].unique()
    sizes = sorted(df['N'].unique())
    
    x = np.arange(len(sizes))
    width = 0.35
    
    # График времени
    for i, version in enumerate(versions):
        data = df[df['version'] == version]
        times = []
        for n in sizes:
            matching = data[data['N'] == n]
            if len(matching) > 0:
                # Use mean to average multiple runs for the same N
                times.append(matching['time_ms'].mean())
            else:
                times.append(0)
        offset = (i - len(versions)/2 + 0.5) * width / len(versions)
        ax1.bar(x + offset, times, width/len(versions), label=version, alpha=0.8)
    
    ax1.set_xlabel('Размер матрицы N', fontsize=13, fontweight='bold')
    ax1.set_ylabel('Время выполнения (мс)', fontsize=13, fontweight='bold')
    ax1.set_title('Сравнение версий с padding и без', fontsize=14, fontweight='bold')
    ax1.set_xticks(x)
    ax1.set_xticklabels(sizes)
    ax1.legend(fontsize=11)
    ax1.grid(True, alpha=0.3, linestyle='--', axis='y')
    
    # График производительности
    for i, version in enumerate(versions):
        data = df[df['version'] == version]
        gflops = []
        for n in sizes:
            matching = data[data['N'] == n]
            if len(matching) > 0:
                # Use mean to average multiple runs for the same N
                gflops.append(matching['gflops'].mean())
            else:
                gflops.append(0)
        offset = (i - len(versions)/2 + 0.5) * width / len(versions)
        ax2.bar(x + offset, gflops, width/len(versions), label=version, alpha=0.8)
    
    ax2.set_xlabel('Размер матрицы N', fontsize=13, fontweight='bold')
    ax2.set_ylabel('Производительность (GFLOPS)', fontsize=13, fontweight='bold')
    ax2.set_title('Производительность: padding vs без padding', fontsize=14, fontweight='bold')
    ax2.set_xticks(x)
    ax2.set_xticklabels(sizes)
    ax2.legend(fontsize=11)
    ax2.grid(True, alpha=0.3, linestyle='--', axis='y')
    
    plt.tight_layout()
    plt.savefig('bank_conflicts_analysis.png', dpi=300, bbox_inches='tight')
    print("  Сохранен график: bank_conflicts_analysis.png")
    plt.close()

def plot_block_size_analysis():
    """Анализ Task 2b: зависимость от размера блока"""
    print("Создание графика анализа block_size...")
    
    df = load_results('task2b', 'task2b_results.csv')
    if df is None or len(df) == 0:
        print("  Файл task2b_results.csv не найден или пуст")
        return
    
    # Фильтруем только streams варианты
    df_streams = df[df['variant'] == 'streams'].copy()
    if len(df_streams) == 0:
        print("  Нет данных streams в task2b_results.csv")
        return
    
    # param1 для streams = num_streams, param2 = block_size
    df_streams['num_streams'] = df_streams['param1'].astype(int)
    df_streams['block_size'] = df_streams['param2'].astype(int)
    
    # Фильтруем по фиксированному num_streams=4 для анализа block_size
    original_len = len(df_streams)
    df_filtered = df_streams[df_streams['num_streams'] == 4]
    if len(df_filtered) > 0:
        df = df_filtered
        print(f"  Используем данные с num_streams=4 ({len(df)} из {original_len} записей)")
    else:
        print(f"  Нет данных с num_streams=4, используем все данные")
        df = df_streams
    
    if len(df) == 0:
        print("  Нет данных для анализа block_size")
        return
    
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 7))
    
    block_sizes = sorted(df['block_size'].unique())
    sizes = sorted(df['N'].unique())
    
    # График времени от размера блока
    for size in sizes:
        data = df[df['N'] == size]
        times = []
        for bs in block_sizes:
            matching = data[data['block_size'] == bs]
            if len(matching) > 0:
                times.append(matching['time_ms'].mean())
            else:
                times.append(np.nan)
        ax1.plot(block_sizes, times, marker='o', linewidth=2.5, markersize=10, 
                label=f'N={size}', markerfacecolor='white', markeredgewidth=2)
    
    ax1.set_xlabel('Размер блока (block_size)', fontsize=13, fontweight='bold')
    ax1.set_ylabel('Время выполнения (мс)', fontsize=13, fontweight='bold')
    ax1.set_title('Время выполнения vs Размер блока (num_streams=4)', fontsize=14, fontweight='bold')
    ax1.legend(fontsize=11, loc='best')
    ax1.grid(True, alpha=0.3, linestyle='--')
    
    # График производительности от размера блока
    for size in sizes:
        data = df[df['N'] == size]
        gflops = []
        for bs in block_sizes:
            matching = data[data['block_size'] == bs]
            if len(matching) > 0:
                gflops.append(matching['gflops'].mean())
            else:
                gflops.append(np.nan)
        ax2.plot(block_sizes, gflops, marker='s', linewidth=2.5, markersize=10, 
                label=f'N={size}', markerfacecolor='white', markeredgewidth=2)
    
    ax2.set_xlabel('Размер блока (block_size)', fontsize=13, fontweight='bold')
    ax2.set_ylabel('Производительность (GFLOPS)', fontsize=13, fontweight='bold')
    ax2.set_title('Производительность vs Размер блока (num_streams=4)', fontsize=14, fontweight='bold')
    ax2.legend(fontsize=11, loc='best')
    ax2.grid(True, alpha=0.3, linestyle='--')
    
    plt.tight_layout()
    plt.savefig('block_size_analysis.png', dpi=300, bbox_inches='tight')
    print("  Сохранен график: block_size_analysis.png")
    plt.close()

def plot_size_scalability(df):
    """График масштабируемости по размеру матрицы"""
    print("Создание графика масштабируемости...")
    
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 7))
    
    methods = df['task'].unique()
    sizes = sorted(df['N'].unique())
    
    # График времени с логарифмической шкалой
    for method in methods:
        data = df[df['task'] == method]
        times = []
        for n in sizes:
            matching = data[data['N'] == n]
            if len(matching) > 0:
                # Use mean to average multiple runs for the same N
                times.append(matching['time_ms'].mean())
            else:
                times.append(np.nan)
        ax1.plot(sizes, times, marker='o', linewidth=2.5, markersize=10, 
                label=method, markerfacecolor='white', markeredgewidth=2)
    
    ax1.set_xlabel('Размер матрицы N', fontsize=13, fontweight='bold')
    ax1.set_ylabel('Время выполнения (мс)', fontsize=13, fontweight='bold')
    ax1.set_title('Масштабируемость: Время выполнения', fontsize=14, fontweight='bold')
    ax1.legend(fontsize=11, loc='best')
    ax1.grid(True, alpha=0.3, linestyle='--')
    ax1.set_xscale('log', base=2)
    ax1.set_yscale('log')
    
    # График производительности
    for method in methods:
        data = df[df['task'] == method]
        gflops = []
        for n in sizes:
            matching = data[data['N'] == n]
            if len(matching) > 0:
                # Use mean to average multiple runs for the same N
                gflops.append(matching['gflops'].mean())
            else:
                gflops.append(np.nan)
        ax2.plot(sizes, gflops, marker='s', linewidth=2.5, markersize=10, 
                label=method, markerfacecolor='white', markeredgewidth=2)
    
    ax2.set_xlabel('Размер матрицы N', fontsize=13, fontweight='bold')
    ax2.set_ylabel('Производительность (GFLOPS)', fontsize=13, fontweight='bold')
    ax2.set_title('Масштабируемость: Производительность', fontsize=14, fontweight='bold')
    ax2.legend(fontsize=11, loc='best')
    ax2.grid(True, alpha=0.3, linestyle='--')
    ax2.set_xscale('log', base=2)
    
    plt.tight_layout()
    plt.savefig('scalability_analysis.png', dpi=300, bbox_inches='tight')
    print("  Сохранен график: scalability_analysis.png")
    plt.close()

def plot_task2b_variants_comparison():
    """Сравнение всех вариантов Task 2b"""
    print("Создание графика сравнения вариантов Task 2b...")
    
    df = load_results('task2b', 'task2b_results.csv')
    if df is None or len(df) == 0:
        print("  Файл task2b_results.csv не найден или пуст")
        return
    
    # Для каждого варианта берем лучший результат (по GFLOPS)
    # Для streams берем лучший по num_streams
    variants = df['variant'].unique()
    
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 7))
    
    sizes = sorted(df['N'].unique())
    
    # График времени выполнения
    for variant in variants:
        data = df[df['variant'] == variant]
        times = []
        gflops_list = []
        
        for n in sizes:
            matching = data[data['N'] == n]
            if len(matching) > 0:
                # Берем лучший результат (максимальный GFLOPS = минимальное время)
                best = matching.loc[matching['gflops'].idxmax()]
                times.append(best['time_ms'])
                gflops_list.append(best['gflops'])
            else:
                times.append(np.nan)
                gflops_list.append(np.nan)
        
        ax1.plot(sizes, times, marker='o', linewidth=2.5, markersize=10, 
                label=variant, markerfacecolor='white', markeredgewidth=2)
        ax2.plot(sizes, gflops_list, marker='s', linewidth=2.5, markersize=10, 
                label=variant, markerfacecolor='white', markeredgewidth=2)
    
    ax1.set_xlabel('Размер матрицы N', fontsize=13, fontweight='bold')
    ax1.set_ylabel('Время выполнения (мс)', fontsize=13, fontweight='bold')
    ax1.set_title('Task 2b: Сравнение всех вариантов (время)', fontsize=14, fontweight='bold')
    ax1.legend(fontsize=11, loc='best')
    ax1.grid(True, alpha=0.3, linestyle='--')
    ax1.set_xscale('log', base=2)
    ax1.set_yscale('log')
    
    ax2.set_xlabel('Размер матрицы N', fontsize=13, fontweight='bold')
    ax2.set_ylabel('Производительность (GFLOPS)', fontsize=13, fontweight='bold')
    ax2.set_title('Task 2b: Сравнение всех вариантов (производительность)', fontsize=14, fontweight='bold')
    ax2.legend(fontsize=11, loc='best')
    ax2.grid(True, alpha=0.3, linestyle='--')
    ax2.set_xscale('log', base=2)
    
    plt.tight_layout()
    plt.savefig('task2b_variants_comparison.png', dpi=300, bbox_inches='tight')
    print("  Сохранен график: task2b_variants_comparison.png")
    plt.close()

def create_summary_table(df):
    """Создает сводную таблицу результатов"""
    print("\n=== Сводная таблица результатов ===")
    summary = df.groupby(['task', 'N']).agg({
        'time_ms': 'mean',
        'gflops': 'mean'
    }).round(2)
    print(summary)
    print()

def main():
    print("=" * 60)
    print("Анализ результатов CUDA Matrix Multiplication")
    print("=" * 60)
    print()
    
    # Загрузить все результаты
    df = load_all_results()
    
    if df is None or len(df) == 0:
        print("Результаты не найдены!")
        print("Убедитесь, что программы были запущены и создали CSV файлы.")
        print("\nДля создания результатов запустите:")
        print("  cd task1 && make && ./gemm 1024")
        print("  cd task2a && make && ./gemm 1024")
        print("  и т.д.")
        return
    
    print(f"Методы: {', '.join(df['task'].unique())}")
    print(f"Размеры матриц: {sorted(df['N'].unique())}\n")
    
    # Создать графики
    plot_comparison_by_size(df)
    plot_size_scalability(df)
    plot_streams_analysis()
    plot_bank_conflicts()
    plot_block_size_analysis()  # Дополнительный график для анализа block_size
    plot_task2b_variants_comparison()  # Сравнение всех вариантов task2b
    
    # Сводная таблица
    create_summary_table(df)
    
    print("=" * 60)
    print("Анализ завершен!")
    print("=" * 60)
    print("\nСозданные графики:")
    print("  - performance_comparison.png")
    print("  - scalability_analysis.png")
    print("  - streams_analysis.png")
    print("  - block_size_analysis.png")
    print("  - bank_conflicts_analysis.png")
    print("  - task2b_variants_comparison.png")
    print("\nДля создания PDF из графиков используйте:")
    print("  convert *.png gemm.pdf")
    print("  (требует ImageMagick)")

if __name__ == "__main__":
    main()

