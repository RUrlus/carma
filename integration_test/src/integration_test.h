#ifndef INTEGRATION_TEST_SRC_INTEGRATION_TEST_H_
#define INTEGRATION_TEST_SRC_INTEGRATION_TEST_H_

#include <pybind11/numpy.h>
#include <pybind11/pybind11.h>

#include <carma>
#include <armadillo>

namespace py = pybind11;

void bind_test_is_f_contiguous(py::module& m);

#endif  // INTEGRATION_TEST_SRC_INTEGRATION_TEST_H_
