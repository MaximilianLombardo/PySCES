"""
Exploration of MLX acceleration for ARACNe.

This script explores how MLX can be used to accelerate the ARACNe algorithm
on Apple Silicon hardware.

MLX is a framework for machine learning on Apple Silicon, providing
efficient array operations and automatic differentiation.
"""

import numpy as np
import pandas as pd
import anndata as ad
import time
import logging
import matplotlib.pyplot as plt
import seaborn as sns
import sys
import os

# Try to import MLX
try:
    import mlx.core as mx
    HAS_MLX = True
except ImportError:
    HAS_MLX = False
    print("MLX not found. Install with: pip install mlx")
    print("Note: MLX only works on Apple Silicon hardware.")

# Add the src directory to the Python path
sys.path.append(os.path.join(os.path.dirname(__file__), '..', 'pysces', 'src'))

# Import pysces modules
from pysces.aracne.core import ARACNe

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def create_synthetic_dataset(n_cells, n_genes, n_tfs):
    """Create a synthetic dataset for benchmarking."""
    # Create random count data
    X = np.random.randint(0, 10, size=(n_cells, n_genes))

    # Create gene names with some TFs
    gene_names = [f"gene_{i}" for i in range(n_genes)]
    tf_names = [f"TF_{i}" for i in range(n_tfs)]
    gene_names[:n_tfs] = tf_names  # First n_tfs genes are TFs

    # Create cell names
    cell_names = [f"cell_{i}" for i in range(n_cells)]

    # Create AnnData object
    adata = ad.AnnData(
        X=X,
        obs=pd.DataFrame(index=cell_names),
        var=pd.DataFrame(index=gene_names)
    )

    return adata, tf_names, X

def calculate_correlation_numpy(x, y):
    """Calculate Pearson correlation using NumPy."""
    return np.corrcoef(x, y)[0, 1]

def calculate_correlation_mlx(x, y):
    """Calculate Pearson correlation using MLX."""
    if not HAS_MLX:
        return calculate_correlation_numpy(x, y)
    
    # Convert to MLX arrays
    x_mx = mx.array(x)
    y_mx = mx.array(y)
    
    # Calculate means
    mean_x = mx.mean(x_mx)
    mean_y = mx.mean(y_mx)
    
    # Calculate covariance and variances
    diff_x = x_mx - mean_x
    diff_y = y_mx - mean_y
    
    cov = mx.mean(diff_x * diff_y)
    var_x = mx.mean(diff_x * diff_x)
    var_y = mx.mean(diff_y * diff_y)
    
    # Calculate correlation
    correlation = cov / mx.sqrt(var_x * var_y)
    
    # Convert back to NumPy
    return float(correlation)

def calculate_mi_matrix_mlx(expr_matrix, tf_indices):
    """Calculate mutual information matrix using MLX."""
    if not HAS_MLX:
        return None
    
    n_samples, n_genes = expr_matrix.shape
    n_tfs = len(tf_indices)
    
    # Convert to MLX array
    expr_mx = mx.array(expr_matrix)
    
    # Initialize MI matrix
    mi_matrix = np.zeros((n_tfs, n_genes), dtype=np.float64)
    
    # Calculate MI for each TF-gene pair
    for i, tf_idx in enumerate(tf_indices):
        x = expr_mx[:, tf_idx]
        
        for j in range(n_genes):
            # Skip self-interactions
            if tf_idx == j:
                continue
            
            # Get gene expression
            y = expr_mx[:, j]
            
            # Calculate correlation as a proxy for MI
            correlation = calculate_correlation_mlx(expr_matrix[:, tf_idx], expr_matrix[:, j])
            mi = abs(correlation)
            
            mi_matrix[i, j] = mi
    
    return mi_matrix

def benchmark_correlation():
    """Benchmark correlation calculation with NumPy and MLX."""
    if not HAS_MLX:
        print("MLX not available. Skipping benchmark.")
        return
    
    # Define array sizes
    sizes = [100, 1000, 10000, 100000]
    
    # Run benchmarks
    results = []
    
    for size in sizes:
        # Create random arrays
        x = np.random.randn(size)
        y = np.random.randn(size)
        
        # Benchmark NumPy
        start_time = time.time()
        for _ in range(100):
            calculate_correlation_numpy(x, y)
        numpy_time = (time.time() - start_time) / 100
        
        # Benchmark MLX
        start_time = time.time()
        for _ in range(100):
            calculate_correlation_mlx(x, y)
        mlx_time = (time.time() - start_time) / 100
        
        results.append({
            "size": size,
            "numpy_time": numpy_time,
            "mlx_time": mlx_time,
            "speedup": numpy_time / mlx_time
        })
    
    # Convert results to DataFrame
    results_df = pd.DataFrame(results)
    
    # Print results
    print("\nCorrelation Benchmark Results:")
    print(results_df)
    
    # Plot results
    plt.figure(figsize=(12, 6))
    
    # Plot times
    plt.subplot(1, 2, 1)
    plt.plot(results_df["size"], results_df["numpy_time"], marker='o', label="NumPy")
    plt.plot(results_df["size"], results_df["mlx_time"], marker='x', label="MLX")
    plt.title("Correlation Calculation Time")
    plt.xlabel("Array Size")
    plt.ylabel("Time (seconds)")
    plt.xscale("log")
    plt.yscale("log")
    plt.legend()
    
    # Plot speedup
    plt.subplot(1, 2, 2)
    plt.plot(results_df["size"], results_df["speedup"], marker='o')
    plt.title("MLX Speedup Factor")
    plt.xlabel("Array Size")
    plt.ylabel("Speedup (NumPy / MLX)")
    plt.xscale("log")
    plt.axhline(y=1.0, color='r', linestyle='--')
    
    plt.tight_layout()
    plt.savefig("correlation_benchmark.png")
    plt.show()
    
    return results_df

def benchmark_mi_matrix():
    """Benchmark MI matrix calculation with NumPy and MLX."""
    if not HAS_MLX:
        print("MLX not available. Skipping benchmark.")
        return
    
    # Define dataset sizes
    dataset_sizes = [
        (100, 100, 10),   # Small dataset
        (500, 200, 20),   # Medium dataset
        (1000, 500, 50),  # Large dataset
    ]
    
    # Run benchmarks
    results = []
    
    for n_cells, n_genes, n_tfs in dataset_sizes:
        # Create synthetic dataset
        adata, tf_names, expr_matrix = create_synthetic_dataset(n_cells, n_genes, n_tfs)
        tf_indices = np.arange(n_tfs)
        
        # Benchmark NumPy
        start_time = time.time()
        mi_matrix_numpy = np.zeros((n_tfs, n_genes), dtype=np.float64)
        for i, tf_idx in enumerate(tf_indices):
            for j in range(n_genes):
                if tf_idx == j:
                    continue
                correlation = calculate_correlation_numpy(expr_matrix[:, tf_idx], expr_matrix[:, j])
                mi_matrix_numpy[i, j] = abs(correlation)
        numpy_time = time.time() - start_time
        
        # Benchmark MLX
        start_time = time.time()
        mi_matrix_mlx = calculate_mi_matrix_mlx(expr_matrix, tf_indices)
        mlx_time = time.time() - start_time
        
        results.append({
            "n_cells": n_cells,
            "n_genes": n_genes,
            "n_tfs": n_tfs,
            "numpy_time": numpy_time,
            "mlx_time": mlx_time,
            "speedup": numpy_time / mlx_time
        })
    
    # Convert results to DataFrame
    results_df = pd.DataFrame(results)
    
    # Print results
    print("\nMI Matrix Benchmark Results:")
    print(results_df)
    
    # Plot results
    plt.figure(figsize=(12, 6))
    
    # Plot times
    plt.subplot(1, 2, 1)
    x = np.arange(len(dataset_sizes))
    width = 0.35
    
    plt.bar(x - width/2, results_df["numpy_time"], width, label="NumPy")
    plt.bar(x + width/2, results_df["mlx_time"], width, label="MLX")
    
    plt.title("MI Matrix Calculation Time")
    plt.xlabel("Dataset Size")
    plt.ylabel("Time (seconds)")
    plt.xticks(x, [f"{c}x{g}" for c, g, _ in dataset_sizes])
    plt.legend()
    
    # Plot speedup
    plt.subplot(1, 2, 2)
    plt.bar(x, results_df["speedup"])
    plt.title("MLX Speedup Factor")
    plt.xlabel("Dataset Size")
    plt.ylabel("Speedup (NumPy / MLX)")
    plt.xticks(x, [f"{c}x{g}" for c, g, _ in dataset_sizes])
    plt.axhline(y=1.0, color='r', linestyle='--')
    
    plt.tight_layout()
    plt.savefig("mi_matrix_benchmark.png")
    plt.show()
    
    return results_df

def main():
    """Run all benchmarks and explorations."""
    print("Exploring MLX acceleration for ARACNe")
    
    if not HAS_MLX:
        print("MLX not available. Install with: pip install mlx")
        print("Note: MLX only works on Apple Silicon hardware.")
        return
    
    # Benchmark correlation calculation
    print("\n=== Benchmarking Correlation Calculation ===")
    correlation_results = benchmark_correlation()
    
    # Benchmark MI matrix calculation
    print("\n=== Benchmarking MI Matrix Calculation ===")
    mi_matrix_results = benchmark_mi_matrix()
    
    print("\n=== Exploration Complete ===")
    print("See correlation_benchmark.png and mi_matrix_benchmark.png for results.")

if __name__ == "__main__":
    main()
