#include <pybind11/pybind11.h>
// include numpy header for usage of array_t
#include <pybind11/numpy.h>

namespace py = pybind11;

#include "test_nparray.h"
#include "test_arr_to_mat.h"
#include "test_mat_to_arr.h"
#include "test_type_caster.h"

PYBIND11_MODULE(test_carma, m) {
    // arr_to_mat
    bind_test_arr_to_row(m);
    bind_test_arr_to_col(m);
    bind_test_arr_to_cube(m);
    bind_test_arr_to_mat_1d(m);
    bind_test_arr_to_mat_long(m);
    bind_test_arr_to_mat_double(m);
    bind_test_arr_to_mat_double_copy(m);
    bind_test_to_arma_mat(m);
    bind_test_to_arma_col(m);
    bind_test_to_arma_row(m);
    bind_test_to_arma_cube(m);
    // mat_to_arr
    bind_test_mat_to_arr(m);
    bind_test_row_to_arr(m);
    bind_test_col_to_arr(m);
    bind_test_cube_to_arr(m);
    bind_test_mat_to_arr_return(m);
    bind_test_row_to_arr_return(m);
    bind_test_col_to_arr_return(m);
    bind_test_cube_to_arr_return(m);
    bind_test_mat_to_arr_plus_one(m);
    bind_test_row_to_arr_plus_one(m);
    bind_test_col_to_arr_plus_one(m);
    bind_test_cube_to_arr_plus_one(m);
    // to_numpy
    bind_test_to_numpy_mat(m);
    bind_test_to_numpy_row(m);
    bind_test_to_numpy_col(m);
    bind_test_to_numpy_cube(m);
    // update_array
    bind_test_update_array_mat(m);
    bind_test_update_array_row(m);
    bind_test_update_array_col(m);
    bind_test_update_array_cube(m);
    // nparray
    bind_test_is_owndata(m);
    bind_test_is_aligned(m);
    bind_test_is_writable(m);
    bind_test_flat_reference(m);
    bind_test_is_f_contiguous(m);
    bind_test_is_c_contiguous(m);
    bind_test_flat_reference(m);
    bind_test_mutable_flat_reference(m);
    // type caster
    bind_test_tc_in_mat(m);
    bind_test_tc_in_row(m);
    bind_test_tc_in_col(m);
    bind_test_tc_in_cube(m);
    bind_test_tc_out_mat(m);
    bind_test_tc_out_mat_rvalue(m);
    bind_test_tc_out_row(m);
}
