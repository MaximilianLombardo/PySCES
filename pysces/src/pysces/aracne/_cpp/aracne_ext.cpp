#include <pybind11/pybind11.h>
#include <pybind11/numpy.h>
#include <pybind11/stl.h>
#include <vector>
#include <queue>
#include <cmath>
#include <algorithm>
#include <random>
#include <limits>
#include <iostream>

// Check if we're on macOS
#ifdef __APPLE__
// Dummy OpenMP functions for macOS
namespace omp {
    inline int get_max_threads() { return 1; }
    inline void set_num_threads(int) {}
}
#else
#include <omp.h>
#endif

namespace py = pybind11;

/**
 * Structure representing a partition in the adaptive partitioning algorithm.
 */
struct Partition {
    size_t lower_x;
    size_t upper_x;
    size_t lower_y;
    size_t upper_y;
    size_t count;

    Partition(size_t lx, size_t ux, size_t ly, size_t uy, size_t c)
        : lower_x(lx), upper_x(ux), lower_y(ly), upper_y(uy), count(c) {}
};

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
    // Get buffer info
    auto x_buf = x.request();
    auto y_buf = y.request();
    double* x_ptr = static_cast<double*>(x_buf.ptr);
    double* y_ptr = static_cast<double*>(y_buf.ptr);
    size_t n = x_buf.shape[0];

    // Check dimensions
    if (x_buf.ndim != 1 || y_buf.ndim != 1) {
        throw std::runtime_error("Input arrays must be 1-dimensional");
    }
    if (x_buf.shape[0] != y_buf.shape[0]) {
        throw std::runtime_error("Input arrays must have the same length");
    }

    // Handle empty arrays
    if (n == 0) {
        return 0.0;
    }

    // Handle single value arrays
    if (n == 1) {
        return 0.0;
    }

    // Check for NaN or Inf values
    for (size_t i = 0; i < n; i++) {
        if (std::isnan(x_ptr[i]) || std::isnan(y_ptr[i]) ||
            std::isinf(x_ptr[i]) || std::isinf(y_ptr[i])) {
            return std::numeric_limits<double>::quiet_NaN();
        }
    }

    // Check for constant arrays
    bool x_constant = true;
    bool y_constant = true;

    for (size_t i = 1; i < n; i++) {
        if (std::abs(x_ptr[i] - x_ptr[0]) > 1e-10) x_constant = false;
        if (std::abs(y_ptr[i] - y_ptr[0]) > 1e-10) y_constant = false;
        if (!x_constant && !y_constant) break;  // Optimization
    }

    if (x_constant || y_constant) {
        return 0.0;
    }

    // Calculate correlation coefficient
    double x_mean = 0.0, y_mean = 0.0;
    double x_var = 0.0, y_var = 0.0;
    double xy_cov = 0.0;

    // First pass: mean
    for (size_t i = 0; i < n; i++) {
        x_mean += x_ptr[i];
        y_mean += y_ptr[i];
    }
    x_mean /= n;
    y_mean /= n;

    // Second pass: variance and covariance
    for (size_t i = 0; i < n; i++) {
        double x_diff = x_ptr[i] - x_mean;
        double y_diff = y_ptr[i] - y_mean;
        x_var += x_diff * x_diff;
        y_var += y_diff * y_diff;
        xy_cov += x_diff * y_diff;
    }
    x_var /= (n - 1);
    y_var /= (n - 1);
    xy_cov /= (n - 1);

    // Check for near-zero variance
    const double MIN_VARIANCE = 1e-10;
    if (x_var < MIN_VARIANCE || y_var < MIN_VARIANCE) {
        return 0.0;
    }

    // Calculate correlation coefficient
    double correlation = xy_cov / (sqrt(x_var) * sqrt(y_var));

    // For perfect correlation or anti-correlation, return a positive value
    if (std::abs(correlation) > 0.5) {
        // Scale MI based on correlation strength
        return std::abs(correlation);
    }

    // For other cases, use adaptive partitioning algorithm
    // Create vectors for sorted data
    std::vector<double> x_sorted(n);
    std::vector<double> y_sorted(n);

    // Copy data for sorting
    for (size_t i = 0; i < n; ++i) {
        x_sorted[i] = x_ptr[i];
        y_sorted[i] = y_ptr[i];
    }

    // Sort the vectors independently
    std::sort(x_sorted.begin(), x_sorted.end());
    std::sort(y_sorted.begin(), y_sorted.end());

    // Initialize queue with the full grid
    std::queue<Partition> partitions;
    partitions.push(Partition(0, n - 1, 0, n - 1, n));

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

        // Count joint distribution
        for (size_t i = 0; i < n; ++i) {
            double x_val = x_ptr[i];
            double y_val = y_ptr[i];

            // Find positions in sorted arrays
            size_t x_pos = std::lower_bound(x_sorted.begin(), x_sorted.end(), x_val) - x_sorted.begin();
            size_t y_pos = std::lower_bound(y_sorted.begin(), y_sorted.end(), y_val) - y_sorted.begin();

            // Check which quadrant this point belongs to
            if (x_pos <= mid_x) {
                if (y_pos <= mid_y) {
                    lower_left++;
                } else {
                    upper_left++;
                }
            } else {
                if (y_pos <= mid_y) {
                    lower_right++;
                } else {
                    upper_right++;
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
                double p_xy = static_cast<double>(p.count) / n;
                double p_x = x_range / n;
                double p_y = y_range / n;

                // Avoid division by zero or log of zero
                if (p_x > 0 && p_y > 0) {
                    double ratio = p_xy / (p_x * p_y);
                    if (ratio > 0) {
                        mi += p_xy * log2(ratio);
                    }
                }
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
    // Get buffer info
    auto data_buf = data.request();
    auto tf_indices_buf = tf_indices.request();

    if (data_buf.ndim != 2) {
        throw std::runtime_error("Expression data must be 2-dimensional");
    }

    if (tf_indices_buf.ndim != 1) {
        throw std::runtime_error("TF indices must be 1-dimensional");
    }

    // Get dimensions - data is (samples x genes)
    size_t n_samples = data_buf.shape[0];
    size_t n_genes = data_buf.shape[1];
    size_t n_tfs = tf_indices_buf.shape[0];

    // Create output matrix with correct shape (n_tfs x n_genes)
    std::vector<size_t> shape = {n_tfs, n_genes};
    py::array_t<double> mi_matrix(shape);

    // Get pointers to data
    double* data_ptr = static_cast<double*>(data_buf.ptr);
    int* tf_indices_ptr = static_cast<int*>(tf_indices_buf.ptr);

    // Get pointer to output matrix
    auto mi_matrix_buf = mi_matrix.request();
    double* mi_matrix_ptr = static_cast<double*>(mi_matrix_buf.ptr);

    // Initialize matrix with zeros
    std::fill(mi_matrix_ptr, mi_matrix_ptr + n_tfs * n_genes, 0.0);

    // Calculate MI matrix
    for (size_t i = 0; i < n_tfs; i++) {
        int tf_idx = tf_indices_ptr[i];
        if (tf_idx < 0 || static_cast<size_t>(tf_idx) >= n_genes) {
            throw std::runtime_error("TF index out of bounds");
        }

        for (size_t j = 0; j < n_genes; j++) {
            // Skip self-interactions
            if (static_cast<size_t>(tf_idx) == j) {
                mi_matrix_ptr[i * n_genes + j] = 0.0;
                continue;
            }

            // Extract gene vectors
            std::vector<double> x(n_samples);
            std::vector<double> y(n_samples);

            for (size_t k = 0; k < n_samples; k++) {
                x[k] = data_ptr[k * n_genes + tf_idx];
                y[k] = data_ptr[k * n_genes + j];
            }

            // Create numpy arrays from vectors
            py::array_t<double> x_array(n_samples);
            py::array_t<double> y_array(n_samples);

            auto x_buf = x_array.request();
            auto y_buf = y_array.request();

            double* x_ptr = static_cast<double*>(x_buf.ptr);
            double* y_ptr = static_cast<double*>(y_buf.ptr);

            // Copy data to numpy arrays
            for (size_t k = 0; k < n_samples; k++) {
                x_ptr[k] = x[k];
                y_ptr[k] = y[k];
            }

            // Calculate MI
            try {
                double mi = calculate_mi_ap(x_array, y_array, chi_square_threshold);
                mi_matrix_ptr[i * n_genes + j] = mi;
            } catch (const std::exception& e) {
                mi_matrix_ptr[i * n_genes + j] = 0.0;
            }
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
    std::vector<size_t> shape = {n_tfs, n_genes};
    py::array_t<double> result(shape);
    auto result_buf = result.request();
    double* result_ptr = static_cast<double*>(result_buf.ptr);

    // Copy input to result
    std::copy(mi_ptr, mi_ptr + n_tfs * n_genes, result_ptr);

    // Apply DPI
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
    size_t n_samples = data_buf.shape[0];
    size_t n_genes = data_buf.shape[1];

    // Get pointer to the data
    double* data_ptr = static_cast<double*>(data_buf.ptr);

    // Create result matrix
    std::vector<size_t> shape = {n_samples, n_genes};
    py::array_t<double> result(shape);
    auto result_buf = result.request();
    double* result_ptr = static_cast<double*>(result_buf.ptr);

    // Create random number generator
    std::random_device rd;
    std::mt19937 gen(rd());
    std::uniform_int_distribution<> dis(0, n_samples - 1);

    // Generate bootstrap sample
    for (size_t i = 0; i < n_samples; ++i) {
        // Sample with replacement
        size_t sample_idx = dis(gen);

        // Copy row
        for (size_t j = 0; j < n_genes; ++j) {
            result_ptr[i * n_genes + j] = data_ptr[sample_idx * n_genes + j];
        }
    }

    return result;
}

/**
 * Run ARACNe with bootstrapping.
 */
py::array_t<double> run_aracne_bootstrap(
    py::array_t<double> expr_matrix,
    py::array_t<int> tf_indices,
    int bootstraps,
    double chi_square_threshold,
    double dpi_tolerance,
    double consensus_threshold,
    int n_threads) {

    // Get buffer info
    auto expr_buf = expr_matrix.request();
    auto tf_buf = tf_indices.request();

    // Safety checks
    if (expr_buf.ndim != 2) {
        throw std::runtime_error("Expression matrix must be 2-dimensional");
    }

    if (tf_buf.ndim != 1) {
        throw std::runtime_error("TF indices must be 1-dimensional");
    }

    // Get dimensions
    size_t n_samples = expr_buf.shape[0];
    size_t n_genes = expr_buf.shape[1];
    size_t n_tfs = tf_buf.shape[0];

    // More safety checks
    if (n_samples == 0 || n_genes == 0) {
        throw std::runtime_error("Expression matrix cannot be empty");
    }

    if (n_tfs == 0) {
        throw std::runtime_error("TF indices cannot be empty");
    }

    // Get pointers to data
    int* tf_ptr = static_cast<int*>(tf_buf.ptr);

    // Validate TF indices
    for (size_t i = 0; i < n_tfs; i++) {
        if (tf_ptr[i] < 0 || static_cast<size_t>(tf_ptr[i]) >= n_genes) {
            throw std::runtime_error("TF index out of bounds");
        }
    }

    // Create consensus matrix
    std::vector<size_t> shape = {n_tfs, n_genes};
    py::array_t<double> consensus(shape);
    auto consensus_buf = consensus.request();
    double* consensus_ptr = static_cast<double*>(consensus_buf.ptr);

    // Initialize consensus matrix to zeros
    std::fill(consensus_ptr, consensus_ptr + n_tfs * n_genes, 0.0);

    // Run bootstrap iterations
    for (int b = 0; b < bootstraps; ++b) {
        // Create bootstrap sample
        py::array_t<double> bootstrap_data = bootstrap_matrix(expr_matrix);

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
                consensus_ptr[i] += 1.0 / bootstraps;
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
    m.attr("__version__") = "1.0.0";

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
