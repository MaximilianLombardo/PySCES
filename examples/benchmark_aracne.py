"""
Benchmark script for ARACNe implementation with and without Numba acceleration.
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

# Add the src directory to the Python path
sys.path.append(os.path.join(os.path.dirname(__file__), '..', 'pysces', 'src'))

# Import ARACNe directly from the core module
import importlib.util

# Import core module
core_path = os.path.join(os.path.dirname(__file__), '..', 'pysces', 'src', 'pysces', 'aracne', 'core.py')
spec = importlib.util.spec_from_file_location("core", core_path)
core = importlib.util.module_from_spec(spec)
spec.loader.exec_module(core)

# Import numba_optimized module
numba_path = os.path.join(os.path.dirname(__file__), '..', 'pysces', 'src', 'pysces', 'aracne', 'numba_optimized.py')
spec = importlib.util.spec_from_file_location("numba_optimized", numba_path)
numba_optimized = importlib.util.module_from_spec(spec)
spec.loader.exec_module(numba_optimized)

# Set up ARACNe class
ARACNe = core.ARACNe

# Make sure the core module can find the numba_optimized module
core.numba_optimized = numba_optimized
core.run_aracne_numba = numba_optimized.run_aracne_numba
core.HAS_NUMBA = numba_optimized.HAS_NUMBA

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

    return adata, tf_names

def benchmark_aracne(n_cells, n_genes, n_tfs, bootstraps=5, use_numba=True, warmup=False):
    """Benchmark ARACNe with and without Numba acceleration."""
    # Create synthetic dataset
    adata, tf_names = create_synthetic_dataset(n_cells, n_genes, n_tfs)

    # Initialize ARACNe
    aracne = ARACNe(
        bootstraps=bootstraps,
        p_value=0.05,
        dpi_tolerance=0.1,
        consensus_threshold=0.5,
        use_numba=use_numba
    )

    # Perform a warmup run for Numba to compile the functions
    if warmup and use_numba:
        logger.info("Performing warmup run for Numba compilation...")
        # Create a smaller dataset for warmup
        warmup_adata, warmup_tf_names = create_synthetic_dataset(50, 50, 5)
        # Run with smaller bootstraps
        aracne.run(warmup_adata, tf_list=warmup_tf_names)
        logger.info("Warmup complete")

    # Run ARACNe and measure time
    start_time = time.time()
    network = aracne.run(adata, tf_list=tf_names)
    elapsed_time = time.time() - start_time

    return elapsed_time, network

def run_benchmarks():
    """Run benchmarks with different dataset sizes and configurations."""
    # Define dataset sizes
    dataset_sizes = [
        (100, 100, 10),   # Small dataset
        (200, 150, 15),   # Medium dataset
        # Larger datasets commented out for faster benchmarking
        # (500, 200, 20),   # Large dataset
        # (1000, 500, 50),  # Very large dataset
    ]

    # Define configurations
    configs = [
        {"use_numba": False, "label": "Python"},
        {"use_numba": True, "label": "Numba"}
    ]

    # Run benchmarks
    results = []

    for n_cells, n_genes, n_tfs in dataset_sizes:
        for config in configs:
            logger.info(f"Benchmarking {config['label']} implementation with {n_cells} cells, {n_genes} genes, {n_tfs} TFs")

            # Add warmup for Numba
            warmup = config["use_numba"] and config["label"] == "Numba"

            elapsed_time, network = benchmark_aracne(
                n_cells=n_cells,
                n_genes=n_genes,
                n_tfs=n_tfs,
                bootstraps=3,  # Even smaller number for quicker testing
                use_numba=config["use_numba"],
                warmup=warmup
            )

            results.append({
                "n_cells": n_cells,
                "n_genes": n_genes,
                "n_tfs": n_tfs,
                "implementation": config["label"],
                "time": elapsed_time,
                "n_edges": len(network["edges"])
            })

            logger.info(f"Elapsed time: {elapsed_time:.2f} seconds")
            logger.info(f"Network has {len(network['regulons'])} regulons and {len(network['edges'])} edges")

    # Convert results to DataFrame
    results_df = pd.DataFrame(results)

    # Print results
    print("\nBenchmark Results:")
    print(results_df)

    # Plot results
    plt.figure(figsize=(12, 6))

    # Plot by dataset size
    plt.subplot(1, 2, 1)
    sns.barplot(
        data=results_df,
        x="n_cells",
        y="time",
        hue="implementation"
    )
    plt.title("ARACNe Performance by Dataset Size")
    plt.xlabel("Number of Cells")
    plt.ylabel("Time (seconds)")

    # Plot speedup
    plt.subplot(1, 2, 2)
    speedup_data = []

    for (n_cells, n_genes, n_tfs) in dataset_sizes:
        python_time = results_df[(results_df["n_cells"] == n_cells) &
                                (results_df["implementation"] == "Python")]["time"].values[0]
        numba_time = results_df[(results_df["n_cells"] == n_cells) &
                               (results_df["implementation"] == "Numba")]["time"].values[0]

        speedup = python_time / numba_time

        speedup_data.append({
            "n_cells": n_cells,
            "speedup": speedup
        })

    speedup_df = pd.DataFrame(speedup_data)

    sns.barplot(
        data=speedup_df,
        x="n_cells",
        y="speedup"
    )
    plt.title("Numba Speedup Factor")
    plt.xlabel("Number of Cells")
    plt.ylabel("Speedup (Python / Numba)")

    plt.tight_layout()
    plt.savefig("benchmark_results.png")
    plt.show()

    return results_df

if __name__ == "__main__":
    run_benchmarks()
