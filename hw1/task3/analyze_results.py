#!/usr/bin/env python3
"""
Script to analyze BLAS comparison results and create visualizations
"""

import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np

def load_and_analyze_results(filename='blas_comparison_results.csv'):
    """Load results and perform analysis"""
    try:
        df = pd.read_csv(filename)
        print(f"Loaded {len(df)} benchmark results from {filename}")
        return df
    except FileNotFoundError:
        print(f"Error: File {filename} not found!")
        return None

def create_performance_plots(df):
    """Create performance comparison plots"""
    
    # Set style
    plt.style.use('seaborn-v0_8')
    fig, axes = plt.subplots(2, 2, figsize=(15, 12))
    fig.suptitle('BLAS Library Performance Comparison', fontsize=16, fontweight='bold')
    
    # 1. GFLOPS by method and matrix size
    ax1 = axes[0, 0]
    for method in df['method'].unique():
        method_data = df[df['method'] == method]
        avg_gflops = method_data.groupby('N')['gflops'].mean()
        ax1.plot(avg_gflops.index, avg_gflops.values, marker='o', label=method, linewidth=2)
    
    ax1.set_xlabel('Matrix Size (N)')
    ax1.set_ylabel('Average GFLOPS')
    ax1.set_title('Performance vs Matrix Size')
    ax1.legend()
    ax1.grid(True, alpha=0.3)
    
    # 2. GFLOPS by method and thread count
    ax2 = axes[0, 1]
    for method in df['method'].unique():
        method_data = df[df['method'] == method]
        avg_gflops = method_data.groupby('threads')['gflops'].mean()
        ax2.plot(avg_gflops.index, avg_gflops.values, marker='s', label=method, linewidth=2)
    
    ax2.set_xlabel('Number of Threads')
    ax2.set_ylabel('Average GFLOPS')
    ax2.set_title('Performance vs Thread Count')
    ax2.legend()
    ax2.grid(True, alpha=0.3)
    
    # 3. Heatmap of GFLOPS by method and matrix size
    ax3 = axes[1, 0]
    pivot_data = df.groupby(['method', 'N'])['gflops'].mean().unstack()
    sns.heatmap(pivot_data, annot=True, fmt='.2f', cmap='YlOrRd', ax=ax3)
    ax3.set_title('GFLOPS Heatmap: Method vs Matrix Size')
    ax3.set_xlabel('Matrix Size (N)')
    ax3.set_ylabel('Method')
    
    # 4. Speedup comparison
    ax4 = axes[1, 1]
    sequential_data = df[df['method'] == 'Sequential']
    speedup_data = []
    
    for method in df['method'].unique():
        if method != 'Sequential':
            method_data = df[df['method'] == method]
            merged = pd.merge(method_data, sequential_data, on=['N', 'threads'], suffixes=('_method', '_seq'))
            merged['speedup'] = merged['time_ms_seq'] / merged['time_ms_method']
            avg_speedup = merged.groupby('N')['speedup'].mean()
            speedup_data.append((method, avg_speedup))
    
    for method, speedup in speedup_data:
        ax4.plot(speedup.index, speedup.values, marker='^', label=f'{method} vs Sequential', linewidth=2)
    
    ax4.axhline(y=1, color='black', linestyle='--', alpha=0.5, label='No speedup')
    ax4.set_xlabel('Matrix Size (N)')
    ax4.set_ylabel('Speedup')
    ax4.set_title('Speedup vs Sequential Implementation')
    ax4.legend()
    ax4.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig('blas_performance_analysis.png', dpi=300, bbox_inches='tight')
    print("Performance analysis plot saved as 'blas_performance_analysis.png'")
    
    return fig

def print_summary_statistics(df):
    """Print summary statistics"""
    print("\n" + "="*60)
    print("SUMMARY STATISTICS")
    print("="*60)
    
    # Overall statistics
    print("\nOverall Performance by Method:")
    summary = df.groupby('method')['gflops'].agg(['mean', 'std', 'min', 'max'])
    print(summary.round(2))
    
    # Best performance for each matrix size
    print("\nBest Performance by Matrix Size:")
    for N in sorted(df['N'].unique()):
        n_data = df[df['N'] == N]
        best_idx = n_data['gflops'].idxmax()
        best_result = n_data.loc[best_idx]
        print(f"N={N}: {best_result['method']} with {best_result['gflops']:.2f} GFLOPS "
              f"({best_result['threads']} threads, {best_result['time_ms']:.1f} ms)")
    
    # Thread scaling analysis
    print("\nThread Scaling Analysis:")
    for method in df['method'].unique():
        method_data = df[df['method'] == method]
        if len(method_data['threads'].unique()) > 1:
            scaling = method_data.groupby('threads')['gflops'].mean()
            print(f"\n{method}:")
            for threads, gflops in scaling.items():
                print(f"  {threads:2d} threads: {gflops:.2f} GFLOPS")

def main():
    """Main analysis function"""
    print("BLAS Performance Analysis")
    print("="*40)
    
    # Load data
    df = load_and_analyze_results()
    if df is None:
        return
    
    # Print basic info
    print(f"Methods tested: {', '.join(df['method'].unique())}")
    print(f"Matrix sizes: {sorted(df['N'].unique())}")
    print(f"Thread counts: {sorted(df['threads'].unique())}")
    
    # Create visualizations
    create_performance_plots(df)
    
    # Print summary
    print_summary_statistics(df)
    
    print("\n" + "="*60)
    print("Analysis complete! Check 'blas_performance_analysis.png' for visualizations.")
    print("="*60)

if __name__ == "__main__":
    main()
