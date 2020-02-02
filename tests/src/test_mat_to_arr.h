#include <armadillo>
#include <pybind11/numpy.h>
#include <pybind11/pybind11.h>
namespace py = pybind11;

#include <carma/carma.h>

namespace carma { namespace tests {
    py::array_t<double> test_mat_to_arr_return();
    int test_mat_to_arr(bool copy);
    int test_row_to_arr(bool copy);
    int test_col_to_arr(bool copy);
    int test_cube_to_arr(bool copy);
    int test_to_numpy_mat(bool copy);
    int test_to_numpy_cube(bool copy);
    int test_to_numpy_row(bool copy);
    int test_to_numpy_col(bool copy);
    int test_update_array_mat(py::array_t<double> & arr, int rows);
    int test_update_array_row(py::array_t<double> & arr, int rows);
    int test_update_array_col(py::array_t<double> & arr, int rows);
    int test_update_array_cube(py::array_t<double> & arr, int rows);
} /* tests */ } /* carma */

void bind_test_mat_to_arr(py::module &m);
void bind_test_cube_to_arr(py::module &m);
void bind_test_row_to_arr(py::module &m);
void bind_test_col_to_arr(py::module &m);
void bind_test_mat_to_arr_return(py::module &m);
void bind_test_to_numpy_mat(py::module &m);
void bind_test_to_numpy_cube(py::module &m);
void bind_test_to_numpy_row(py::module &m);
void bind_test_to_numpy_col(py::module &m);
void bind_test_update_array_mat(py::module &m);
void bind_test_update_array_row(py::module &m);
void bind_test_update_array_col(py::module &m);
void bind_test_update_array_cube(py::module &m);
