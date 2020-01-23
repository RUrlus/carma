#include <pybind11/pybind11.h>
// include numpy header for usage of array_t
#include <pybind11/numpy.h>

namespace py = pybind11;

#include "test_nparray.h"
#include "test_converters.h"

PYBIND11_MODULE(_test_fsts_core, m) {
    // arma
    bind_test_arr_to_row(m);
    bind_test_arr_to_col(m);
    bind_test_arr_to_mat_1d(m);
    bind_test_arr_to_mat_long(m);
    bind_test_arr_to_mat_double(m);
    bind_test_arr_to_mat_double_copy(m);
    bind_test_mat_to_arr(m);
    // nparray
    bind_test_is_owndata(m);
    bind_test_is_aligned(m);
    bind_test_is_writable(m);
    bind_test_flat_reference(m);
    bind_test_is_f_contiguous(m);
    bind_test_is_c_contiguous(m);
    bind_test_mutable_flat_reference(m);
}
