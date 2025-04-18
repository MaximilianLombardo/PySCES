/**
 * ARACNe C++ extensions for PySCES
 * 
 * This file contains C++ implementations of performance-critical parts of the ARACNe algorithm.
 */

#include "include/aracne.hpp"
#include <cmath>
#include <algorithm>
#include <random>
#include <queue>
#include <tuple>
#include <unordered_map>
#include <iostream>
#include <limits>

namespace py = pybind11;

/**
 * Helper function to perform chi-square test for adaptive partitioning.
 */
bool chi_square_test(size_t a, size_t b, size_t c, size_t d, double chi_square_threshold) {
    // If total count is too small, don't partition further
    size_t total = a + b + c + d;
    if (total < 8) {
        return false;
    }
    
    // Expected count in each quadrant under independence
    double expected = total / 4.0;
    
    // Calculate chi-square statistic
    double chi_square = 
        pow(a - expected, 2) / expected +
        pow(b - expected, 2) / expected +
        pow(c - expected, 2) / expected +
        pow(d - expected, 2) / expected;
    
    // Return true if significant (should continue partitioning)
    return chi_square >= chi_square_threshold;
}

/**
 * Calculate mutual information between two vectors using adaptive partitioning.
 */
double calculate_mi_ap(py::array_t<double> x, py::array_t<double> y, double chi_square_threshold) {
    // Access the data
    auto x_buf = x.request();
    auto y_buf = y.request();
    
    // Check dimensions
    if (x_buf.ndim != 1 || y_buf.ndim != 1) {
        throw std::runtime_error("Input arrays must be 1-dimensional");
    }
    
    if (x_buf.shape[0] != y_buf.shape[0]) {
        throw std::runtime_error("Input arrays must have the same length");
    }
    
    // Get pointers to the data
    double* x_ptr = static_cast<double*>(x_buf.ptr);
    double* y_ptr = static_cast<double*>(y_buf.ptr);
    size_t n_samples = x_buf.shape[0];
    
    // Create a vector of indices for sorting
    std::vector<size_t> indices(n_samples);
    for (size_t i = 0; i < n_samples; ++i) {
        indices[i] = i;
    }
    
    // Sort indices by x values
    std::sort(indices.begin(), indices.end(), [&](size_t i, size_t j) {
        return x_ptr[i] < x_ptr[j];
    });
    
    // Create sorted vectors
    std::vector<double> x_sorted(n_samples);
    std::vector<double> y_sorted(n_samples);
    for (size_t i = 0; i < n_samples; ++i) {
        x_sorted[i] = x_ptr[indices[i]];
        y_sorted[i] = y_ptr[indices[i]];
    }
    
    // Initialize queue with the full grid
    std::queue<Partition> partitions;
    partitions.push(Partition(0, n_samples - 1, 0, n_samples - 1, n_samples));
    
    // Initialize mutual information
    double mi = 0.0;
    
    // Process partitions
    while (!partitions.empty()) {
        Partition p = partitions.front();
        partitions.pop();
        
        // Calculate midpoints
        size_t mid_x = (p.lower_x + p.upper_x) / 2;
        size_t mid_y = (p.lower_y + p.upper_y) / 2;
        
        // Count points in each quadrant
        size_t upper_right = 0;
        size_t upper_left = 0;
        size_t lower_left = 0;
        size_t lower_right = 0;
        
        for (size_t i = p.lower_x; i <= p.upper_x; ++i) {
            for (size_t j = p.lower_y; j <= p.upper_y; ++j) {
                if (x_sorted[i] <= mid_x) {
                    if (y_sorted[j] <= mid_y) {
                        lower_left++;
                    } else {
                        upper_left++;
                    }
                } else {
                    if (y_sorted[j] <= mid_y) {
                        lower_right++;
                    } else {
                        upper_right++;
                    }
                }
            }
        }
        
        // Check if we should continue partitioning
        if (chi_square_test(upper_right, upper_left, lower_left, lower_right, chi_square_threshold)) {
            // Add new partitions to the queue
            if (upper_right > 0) {
                partitions.push(Partition(mid_x + 1, p.upper_x, mid_y + 1, p.upper_y, upper_right));
            }
            if (upper_left > 0) {
                partitions.push(Partition(p.lower_x, mid_x, mid_y + 1, p.upper_y, upper_left));
            }
            if (lower_left > 0) {
                partitions.push(Partition(p.lower_x, mid_x, p.lower_y, mid_y, lower_left));
            }
            if (lower_right > 0) {
                partitions.push(Partition(mid_x + 1, p.upper_x, p.lower_y, mid_y, lower_right));
            }
        } else {
            // Calculate contribution to mutual information
            double x_range = p.upper_x - p.lower_x + 1;
            double y_range = p.upper_y - p.lower_y + 1;
            
            if (p.count > 0) {
                double p_xy = static_cast<double>(p.count) / n_samples;
                double p_x = x_range / n_samples;
                double p_y = y_range / n_samples;
                
                mi += p_xy * log2(p_xy / (p_x * p_y));
            }
        }
    }
    
    return mi;
}

/**
 * Calculate mutual information matrix for all gene pairs.
 */
py::array_t<double> calculate_mi_matrix(
    py::array_t<double> data, 
    py::array_t<int> tf_indices,
    double chi_square_threshold,
    int n_threads
) {
    // Access the data
    auto data_buf = data.request();
    auto tf_indices_buf = tf_indices.request();
    
    // Check dimensions
    if (data_buf.ndim != 2) {
        throw std::runtime_error("Expression data must be 2-dimensional");
    }
    
    if (tf_indices_buf.ndim != 1) {
        throw std::runtime_error("TF indices must be 1-dimensional");
    }
    
    // Get dimensions
    size_t n_genes = data_buf.shape[0];
    size_t n_samples = data_buf.shape[1];
    size_t n_tfs = tf_indices_buf.shape[0];
    
    // Get pointers to the data
    double* data_ptr = static_cast<double*>(data_buf.ptr);
    int* tf_indices_ptr = static_cast<int*>(tf_indices_buf.ptr);
    
    // Create result matrix
    py::array_t<double> mi_matrix = py::array_t<double>({n_tfs, n_genes});
    auto mi_matrix_buf = mi_matrix.request();
    double* mi_matrix_ptr = static_cast<double*>(mi_matrix_buf.ptr);
    
    // Set number of threads
    if (n_threads <= 0) {
        n_threads = omp_get_max_threads();
    }
    omp_set_num_threads(n_threads);
    
    // Calculate MI for each TF-gene pair
    #pragma omp parallel for schedule(dynamic)
    for (size_t i = 0; i < n_tfs; ++i) {
        int tf_idx = tf_indices_ptr[i];
        
        // Check if TF index is valid
        if (tf_idx < 0 || tf_idx >= static_cast<int>(n_genes)) {
            throw std::runtime_error("Invalid TF index: " + std::to_string(tf_idx));
        }
        
        // Get TF expression vector
        std::vector<double> tf_expr(n_samples);
        for (size_t s = 0; s < n_samples; ++s) {
            tf_expr[s] = data_ptr[tf_idx * n_samples + s];
        }
        
        // Calculate MI for each gene
        for (size_t j = 0; j < n_genes; ++j) {
            // Skip self-interactions
            if (static_cast<int>(j) == tf_idx) {
                mi_matrix_ptr[i * n_genes + j] = 0.0;
                continue;
            }
            
            // Get gene expression vector
            std::vector<double> gene_expr(n_samples);
            for (size_t s = 0; s < n_samples; ++s) {
                gene_expr[s] = data_ptr[j * n_samples + s];
            }
            
            // Calculate MI
            py::array_t<double> x = py::array_t<double>(n_samples);
            py::array_t<double> y = py::array_t<double>(n_samples);
            
            auto x_buf = x.request();
            auto y_buf = y.request();
            
            double* x_ptr = static_cast<double*>(x_buf.ptr);
            double* y_ptr = static_cast<double*>(y_buf.ptr);
            
            for (size_t s = 0; s < n_samples; ++s) {
                x_ptr[s] = tf_expr[s];
                y_ptr[s] = gene_expr[s];
            }
            
            mi_matrix_ptr[i * n_genes + j] = calculate_mi_ap(x, y, chi_square_threshold);
        }
    }
    
    return mi_matrix;
}

/**
 * Apply Data Processing Inequality to a mutual information matrix.
 */
py::array_t<double> apply_dpi(
    py::array_t<double> mi_matrix, 
    double tolerance,
    int n_threads
) {
    // Access the data
    auto mi_buf = mi_matrix.request();
    
    // Check dimensions
    if (mi_buf.ndim != 2) {
        throw std::runtime_error("MI matrix must be 2-dimensional");
    }
    
    // Get dimensions
    size_t n_tfs = mi_buf.shape[0];
    size_t n_genes = mi_buf.shape[1];
    
    // Get pointer to the data
    double* mi_ptr = static_cast<double*>(mi_buf.ptr);
    
    // Create result matrix (copy of input)
    py::array_t<double> result = py::array_t<double>({n_tfs, n_genes});
    auto result_buf = result.request();
    double* result_ptr = static_cast<double*>(result_buf.ptr);
    
    // Copy input to result
    std::copy(mi_ptr, mi_ptr + n_tfs * n_genes, result_ptr);
    
    // Set number of threads
    if (n_threads <= 0) {
        n_threads = omp_get_max_threads();
    }
    omp_set_num_threads(n_threads);
    
    // Apply DPI
    #pragma omp parallel for schedule(dynamic)
    for (size_t i = 0; i < n_tfs; ++i) {
        for (size_t j = 0; j < n_genes; ++j) {
            // Skip zero or very small values
            if (result_ptr[i * n_genes + j] < 1e-10) {
                continue;
            }
            
            // Check all possible mediators
            for (size_t k = 0; k < n_genes; ++k) {
                // Skip if k is i or j
                if (k == i || k == j) {
                    continue;
                }
                
                // Get MI values
                double mi_ij = result_ptr[i * n_genes + j];
                double mi_ik = 0.0;
                double mi_kj = 0.0;
                
                // Find MI(i,k) - could be in either direction
                for (size_t tf_idx = 0; tf_idx < n_tfs; ++tf_idx) {
                    if (result_ptr[tf_idx * n_genes + k] > mi_ik && tf_idx == i) {
                        mi_ik = result_ptr[tf_idx * n_genes + k];
                    }
                    if (result_ptr[tf_idx * n_genes + i] > mi_ik && tf_idx == k) {
                        mi_ik = result_ptr[tf_idx * n_genes + i];
                    }
                }
                
                // Find MI(k,j) - could be in either direction
                for (size_t tf_idx = 0; tf_idx < n_tfs; ++tf_idx) {
                    if (result_ptr[tf_idx * n_genes + j] > mi_kj && tf_idx == k) {
                        mi_kj = result_ptr[tf_idx * n_genes + j];
                    }
                    if (result_ptr[tf_idx * n_genes + k] > mi_kj && tf_idx == j) {
                        mi_kj = result_ptr[tf_idx * n_genes + k];
                    }
                }
                
                // Apply DPI
                if (mi_ij < std::min(mi_ik, mi_kj) - tolerance) {
                    result_ptr[i * n_genes + j] = 0.0;
                    break;
                }
            }
        }
    }
    
    return result;
}

/**
 * Create a bootstrapped sample of a data matrix.
 */
py::array_t<double> bootstrap_matrix(py::array_t<double> data) {
    // Access the data
    auto data_buf = data.request();
    
    // Check dimensions
    if (data_buf.ndim != 2) {
        throw std::runtime_error("Input matrix must be 2-dimensional");
    }
    
    // Get dimensions
    size_t n_genes = data_buf.shape[0];
    size_t n_samples = data_buf.shape[1];
    
    // Get pointer to the data
    double* data_ptr = static_cast<double*>(data_buf.ptr);
    
    // Create result matrix
    py::array_t<double> result = py::array_t<double>({n_genes, n_samples});
    auto result_buf = result.request();
    double* result_ptr = static_cast<double*>(result_buf.ptr);
    
    // Create random number generator
    std::random_device rd;
    std::mt19937 gen(rd());
    std::uniform_int_distribution<> dis(0, n_samples - 1);
    
    // Generate bootstrap sample
    for (size_t j = 0; j < n_samples; ++j) {
        // Sample with replacement
        size_t sample_idx = dis(gen);
        
        // Copy column
        for (size_t i = 0; i < n_genes; ++i) {
            result_ptr[i * n_samples + j] = data_ptr[i * n_samples + sample_idx];
        }
    }
    
    return result;
}

/**
 * Run ARACNe with bootstrapping.
 */
py::array_t<double> run_aracne_bootstrap(
    py::array_t<double> data,
    py::array_t<int> tf_indices,
    int n_bootstraps,
    double chi_square_threshold,
    double dpi_tolerance,
    double consensus_threshold,
    int n_threads
) {
    // Access the data
    auto data_buf = data.request();
    auto tf_indices_buf = tf_indices.request();
    
    // Check dimensions
    if (data_buf.ndim != 2) {
        throw std::runtime_error("Expression data must be 2-dimensional");
    }
    
    if (tf_indices_buf.ndim != 1) {
        throw std::runtime_error("TF indices must be 1-dimensional");
    }
    
    // Get dimensions
    size_t n_genes = data_buf.shape[0];
    size_t n_samples = data_buf.shape[1];
    size_t n_tfs = tf_indices_buf.shape[0];
    
    // Create consensus matrix
    py::array_t<double> consensus = py::array_t<double>({n_tfs, n_genes});
    auto consensus_buf = consensus.request();
    double* consensus_ptr = static_cast<double*>(consensus_buf.ptr);
    
    // Initialize consensus matrix to zeros
    std::fill(consensus_ptr, consensus_ptr + n_tfs * n_genes, 0.0);
    
    // Set number of threads
    if (n_threads <= 0) {
        n_threads = omp_get_max_threads();
    }
    
    // Run bootstrap iterations
    for (int b = 0; b < n_bootstraps; ++b) {
        // Create bootstrap sample
        py::array_t<double> bootstrap_data = bootstrap_matrix(data);
        
        // Calculate MI matrix
        py::array_t<double> mi_matrix = calculate_mi_matrix(
            bootstrap_data, tf_indices, chi_square_threshold, n_threads);
        
        // Apply DPI
        py::array_t<double> pruned_matrix = apply_dpi(
            mi_matrix, dpi_tolerance, n_threads);
        
        // Add to consensus matrix
        auto pruned_buf = pruned_matrix.request();
        double* pruned_ptr = static_cast<double*>(pruned_buf.ptr);
        
        for (size_t i = 0; i < n_tfs * n_genes; ++i) {
            if (pruned_ptr[i] > 0) {
                consensus_ptr[i] += 1.0 / n_bootstraps;
            }
        }
    }
    
    // Apply consensus threshold
    for (size_t i = 0; i < n_tfs * n_genes; ++i) {
        if (consensus_ptr[i] < consensus_threshold) {
            consensus_ptr[i] = 0.0;
        }
    }
    
    return consensus;
}

// Module definition
PYBIND11_MODULE(aracne_ext, m) {
    m.doc() = "C++ extensions for ARACNe algorithm";
    
    m.def("calculate_mi_ap", &calculate_mi_ap, "Calculate mutual information between two vectors using adaptive partitioning",
          py::arg("x"), py::arg("y"), py::arg("chi_square_threshold") = 7.815);
    
    m.def("calculate_mi_matrix", &calculate_mi_matrix, "Calculate mutual information matrix for all gene pairs",
          py::arg("data"), py::arg("tf_indices"), py::arg("chi_square_threshold") = 7.815, py::arg("n_threads") = 0);
    
    m.def("apply_dpi", &apply_dpi, "Apply Data Processing Inequality to MI matrix",
          py::arg("mi_matrix"), py::arg("tolerance") = 0.0, py::arg("n_threads") = 0);
          
    m.def("bootstrap_matrix", &bootstrap_matrix, "Create bootstrapped sample of data matrix",
          py::arg("data"));
          
    m.def("run_aracne_bootstrap", &run_aracne_bootstrap, "Run ARACNe with bootstrapping",
          py::arg("data"), py::arg("tf_indices"), py::arg("n_bootstraps") = 100,
          py::arg("chi_square_threshold") = 7.815, py::arg("dpi_tolerance") = 0.0,
          py::arg("consensus_threshold") = 0.05, py::arg("n_threads") = 0);
}
