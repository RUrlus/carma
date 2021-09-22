#ifndef TESTS_SRC_TEST_ROUNDTRIP_H_
#define TESTS_SRC_TEST_ROUNDTRIP_H_

#include <pybind11/numpy.h>
#include <pybind11/pybind11.h>
#include <armadillo>
#include <carma>

namespace py = pybind11;

namespace carma {
namespace tests {

py::array_t<double> test_mat_roundtrip(py::array_t<double>& arr);
py::array_t<double> test_row_roundtrip(py::array_t<double>& arr);
py::array_t<double> test_col_roundtrip(py::array_t<double>& arr);
py::array_t<double> test_cube_roundtrip(py::array_t<double>& arr);

}  // namespace tests
}  // namespace carma

void bind_test_mat_roundtrip(py::module& m);
void bind_test_row_roundtrip(py::module& m);
void bind_test_col_roundtrip(py::module& m);
void bind_test_cube_roundtrip(py::module& m);

#endif  // TESTS_SRC_TEST_ROUNDTRIP_H_
