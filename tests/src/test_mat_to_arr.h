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
} /* tests */ } /* carma */

void bind_test_mat_to_arr(py::module &m);
void bind_test_cube_to_arr(py::module &m);
void bind_test_row_to_arr(py::module &m);
void bind_test_col_to_arr(py::module &m);
void bind_test_mat_to_arr_return(py::module &m);
