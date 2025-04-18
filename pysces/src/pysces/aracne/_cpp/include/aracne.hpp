#pragma once

#include <pybind11/pybind11.h>
#include <pybind11/numpy.h>
#include <omp.h>
#include <vector>
#include <stdexcept>
#include <cmath>

namespace py = pybind11;

double calculate_mi_ap(
    py::array_t<double> x,
    py::array_t<double> y,
    double chi_square_threshold = 7.815
);

py::array_t<double> calculate_mi_matrix(
    py::array_t<double> data,
    py::array_t<int> tf_indices,
    double chi_square_threshold = 7.815,
    int n_threads = 0
);

py::array_t<double> apply_dpi(
    py::array_t<double> mi_matrix,
    double tolerance = 0.0,
    int n_threads = 0
);
