"""
Create benchmark plots for ARACNe and VIPER optimizations.
"""

import numpy as np
import matplotlib.pyplot as plt
import os

# Create directory for plots
os.makedirs('docs/images', exist_ok=True)

# ARACNe benchmark data
aracne_datasets = [
    "100×100×10",
    "200×150×15",
    "500×200×20",
    "1000×500×50"
]

aracne_python_times = [0.14, 0.54, 1.59, 38.34]
aracne_numba_times = [0.00, 0.01, 0.02, 0.63]
aracne_speedups = [62.92, 89.25, 78.12, 61.26]

# VIPER benchmark data
viper_datasets = [
    "100×100×10",
    "500×200×20",
    "1000×500×50"
]

viper_python_times = [0.04, 0.45, 3.06]
viper_numba_times = [0.00, 0.04, 0.22]
viper_speedups = [10.51, 12.57, 14.15]

# Create execution time comparison plot for ARACNe
plt.figure(figsize=(10, 6))
x = np.arange(len(aracne_datasets))
width = 0.35

plt.bar(x - width/2, aracne_python_times, width, label='Python Implementation')
plt.bar(x + width/2, aracne_numba_times, width, label='Numba Implementation')

plt.yscale('log')
plt.ylabel('Execution Time (seconds, log scale)')
plt.xlabel('Dataset Size (cells × genes × TFs)')
plt.title('ARACNe Execution Time Comparison')
plt.xticks(x, aracne_datasets)
plt.legend()
plt.grid(axis='y', linestyle='--', alpha=0.7)

# Add speedup annotations
for i, speedup in enumerate(aracne_speedups):
    plt.annotate(f'{speedup:.1f}x faster', 
                 xy=(i, max(aracne_python_times[i], aracne_numba_times[i]) * 1.1),
                 ha='center', va='bottom',
                 bbox=dict(boxstyle='round,pad=0.3', fc='yellow', alpha=0.7))

plt.tight_layout()
plt.savefig('docs/images/aracne_benchmark.png', dpi=300)

# Create speedup plot for ARACNe
plt.figure(figsize=(10, 6))
plt.bar(aracne_datasets, aracne_speedups, color='green')
plt.ylabel('Speedup Factor (x times faster)')
plt.xlabel('Dataset Size (cells × genes × TFs)')
plt.title('ARACNe Numba Acceleration Speedup')
plt.grid(axis='y', linestyle='--', alpha=0.7)

for i, speedup in enumerate(aracne_speedups):
    plt.annotate(f'{speedup:.1f}x', 
                 xy=(i, speedup + 2),
                 ha='center', va='bottom')

plt.tight_layout()
plt.savefig('docs/images/aracne_speedup.png', dpi=300)

# Create execution time comparison plot for VIPER
plt.figure(figsize=(10, 6))
x = np.arange(len(viper_datasets))
width = 0.35

plt.bar(x - width/2, viper_python_times, width, label='Python Implementation')
plt.bar(x + width/2, viper_numba_times, width, label='Numba Implementation')

plt.yscale('log')
plt.ylabel('Execution Time (seconds, log scale)')
plt.xlabel('Dataset Size (cells × genes × TFs)')
plt.title('VIPER Execution Time Comparison')
plt.xticks(x, viper_datasets)
plt.legend()
plt.grid(axis='y', linestyle='--', alpha=0.7)

# Add speedup annotations
for i, speedup in enumerate(viper_speedups):
    plt.annotate(f'{speedup:.1f}x faster', 
                 xy=(i, max(viper_python_times[i], viper_numba_times[i]) * 1.1),
                 ha='center', va='bottom',
                 bbox=dict(boxstyle='round,pad=0.3', fc='yellow', alpha=0.7))

plt.tight_layout()
plt.savefig('docs/images/viper_benchmark.png', dpi=300)

# Create speedup plot for VIPER
plt.figure(figsize=(10, 6))
plt.bar(viper_datasets, viper_speedups, color='green')
plt.ylabel('Speedup Factor (x times faster)')
plt.xlabel('Dataset Size (cells × genes × TFs)')
plt.title('VIPER Numba Acceleration Speedup')
plt.grid(axis='y', linestyle='--', alpha=0.7)

for i, speedup in enumerate(viper_speedups):
    plt.annotate(f'{speedup:.1f}x', 
                 xy=(i, speedup + 0.5),
                 ha='center', va='bottom')

plt.tight_layout()
plt.savefig('docs/images/viper_speedup.png', dpi=300)

# Create speedup comparison by dataset size
plt.figure(figsize=(12, 6))

# Combine datasets and speedups
all_datasets = []
all_speedups = []
all_algorithms = []

for dataset, speedup in zip(aracne_datasets, aracne_speedups):
    all_datasets.append(dataset)
    all_speedups.append(speedup)
    all_algorithms.append('ARACNe')

for dataset, speedup in zip(viper_datasets, viper_speedups):
    all_datasets.append(dataset)
    all_speedups.append(speedup)
    all_algorithms.append('VIPER')

# Create a scatter plot
unique_datasets = sorted(set(all_datasets), key=lambda x: int(x.split('×')[0]))
colors = {'ARACNe': 'blue', 'VIPER': 'red'}
markers = {'ARACNe': 'o', 'VIPER': 's'}

for algo in ['ARACNe', 'VIPER']:
    x_vals = [i for i, (d, a) in enumerate(zip(all_datasets, all_algorithms)) if a == algo]
    y_vals = [s for s, a in zip(all_speedups, all_algorithms) if a == algo]
    plt.scatter(x_vals, y_vals, color=colors[algo], marker=markers[algo], s=100, label=algo)

# Connect points with lines
for algo in ['ARACNe', 'VIPER']:
    indices = [i for i, a in enumerate(all_algorithms) if a == algo]
    indices.sort(key=lambda i: all_datasets[i])
    plt.plot([i for i in indices], [all_speedups[i] for i in indices], color=colors[algo], linestyle='-', alpha=0.7)

plt.ylabel('Speedup Factor (x times faster)')
plt.title('Numba Acceleration Speedup by Dataset Size and Algorithm')
plt.xticks(range(len(all_datasets)), all_datasets, rotation=45)
plt.grid(axis='y', linestyle='--', alpha=0.7)
plt.legend()

plt.tight_layout()
plt.savefig('docs/images/speedup_comparison.png', dpi=300)

print("Benchmark plots created in docs/images/")
