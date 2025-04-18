/**
 * ARACNe C++ header file
 * 
 * This file contains declarations for the ARACNe algorithm C++ implementation.
 */

#ifndef ARACNE_HPP
#define ARACNE_HPP

#include <pybind11/pybind11.h>
#include <pybind11/numpy.h>
#include <pybind11/stl.h>
#include <vector>
#include <cmath>
#include <algorithm>
#include <random>
#include <queue>
#include <tuple>
#include <unordered_map>
#include <omp.h>

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
 * Calculate mutual information between two vectors using adaptive partitioning.
 * 
 * @param x First vector (rank-transformed)
 * @param y Second vector (rank-transformed)
 * @param chi_square_threshold Chi-square threshold for partitioning (default: 7.815 for 95% confidence)
 * @return Mutual information value
 */
double calculate_mi_ap(py::array_t<double> x, py::array_t<double> y, double chi_square_threshold = 7.815);

/**
 * Calculate mutual information matrix for all gene pairs.
 * 
 * @param data Expression matrix (genes x cells)
 * @param tf_indices Indices of transcription factors
 * @param chi_square_threshold Chi-square threshold for partitioning
 * @param n_threads Number of threads to use (default: 0 = auto)
 * @return Mutual information matrix (TFs x genes)
 */
py::array_t<double> calculate_mi_matrix(
    py::array_t<double> data, 
    py::array_t<int> tf_indices,
    double chi_square_threshold = 7.815,
    int n_threads = 0
);

/**
 * Apply Data Processing Inequality to a mutual information matrix.
 * 
 * @param mi_matrix Mutual information matrix (TFs x genes)
 * @param tolerance Tolerance for DPI
 * @param n_threads Number of threads to use (default: 0 = auto)
 * @return Pruned adjacency matrix
 */
py::array_t<double> apply_dpi(
    py::array_t<double> mi_matrix, 
    double tolerance = 0.0,
    int n_threads = 0
);

/**
 * Create a bootstrapped sample of a data matrix.
 * 
 * @param data Input data matrix (genes x cells)
 * @return Bootstrapped matrix
 */
py::array_t<double> bootstrap_matrix(py::array_t<double> data);

/**
 * Run ARACNe with bootstrapping.
 * 
 * @param data Expression matrix (genes x cells)
 * @param tf_indices Indices of transcription factors
 * @param n_bootstraps Number of bootstrap iterations
 * @param chi_square_threshold Chi-square threshold for partitioning
 * @param dpi_tolerance Tolerance for DPI
 * @param consensus_threshold Threshold for consensus network
 * @param n_threads Number of threads to use (default: 0 = auto)
 * @return Consensus network matrix
 */
py::array_t<double> run_aracne_bootstrap(
    py::array_t<double> data,
    py::array_t<int> tf_indices,
    int n_bootstraps = 100,
    double chi_square_threshold = 7.815,
    double dpi_tolerance = 0.0,
    double consensus_threshold = 0.05,
    int n_threads = 0
);

/**
 * Helper function to perform chi-square test for adaptive partitioning.
 * 
 * @param a Upper right quadrant count
 * @param b Upper left quadrant count
 * @param c Lower left quadrant count
 * @param d Lower right quadrant count
 * @param chi_square_threshold Threshold for significance
 * @return True if partitioning should continue, false otherwise
 */
bool chi_square_test(size_t a, size_t b, size_t c, size_t d, double chi_square_threshold);

#endif // ARACNE_HPP
