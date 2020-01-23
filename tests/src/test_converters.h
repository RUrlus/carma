#include <pybind11/numpy.h>
#include <pybind11/pybind11.h>
namespace py = pybind11;

#include <carma/carma.h>

namespace carma { namespace tests {

int test_arr_to_mat_double_copy(py::array_t<double> arr);
int test_arr_to_mat_1d(py::array_t<long> & arr, bool copy, bool strict);
int test_arr_to_mat_long(py::array_t<long> & arr, bool copy, bool strict);
int test_arr_to_mat_double(py::array_t<double> & arr, bool copy, bool strict);
int test_arr_to_col(py::array_t<double> & arr, bool copy, bool strict);
int test_arr_to_row(py::array_t<double> & arr, bool copy, bool strict);
py::array_t<double> test_mat_to_arr(py::array_t<double> & arr);

} /* tests */
} /* carma */

void bind_test_arr_to_row(py::module &m);
void bind_test_arr_to_col(py::module &m);
void bind_test_arr_to_mat_1d(py::module &m);
void bind_test_arr_to_mat_long(py::module &m);
void bind_test_arr_to_mat_double(py::module &m);
void bind_test_arr_to_mat_double_copy(py::module &m);
void bind_test_mat_to_arr(py::module &m);
