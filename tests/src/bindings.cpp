
#include <pybind11/pybind11.h>
// include numpy header for usage of array_t
#include <pybind11/numpy.h>

#include <carma>
#include <armadillo>

#include "test_arr_to_mat.h"
#include "test_arraystore.h"
#include "test_mat_to_arr.h"
#include "test_nparray.h"
#include "test_roundtrip.h"
#include "test_type_caster.h"

#include <string>

namespace py = pybind11;

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
    bind_test_mat_to_arr_plus_one(m);
    bind_test_row_to_arr_plus_one(m);
    bind_test_col_to_arr_plus_one(m);
    bind_test_cube_to_arr_plus_one(m);

    // to_numpy
    bind_test_to_numpy_mat(m);
    bind_test_to_numpy_row(m);
    bind_test_to_numpy_col(m);
    bind_test_to_numpy_cube(m);
    bind_test_to_numpy_view_mat(m);
    bind_test_to_numpy_view_row(m);
    bind_test_to_numpy_view_col(m);
    bind_test_to_numpy_view_cube(m);

    // nparray
    bind_test_is_owndata(m);
    bind_test_is_aligned(m);
    bind_test_is_writeable(m);
    bind_test_is_f_contiguous(m);
    bind_test_is_c_contiguous(m);
    bind_test_set_not_owndata(m);
    bind_test_set_not_writeable(m);
    bind_test_is_well_behaved(m);
    bind_test_exception_flow(m);
    bind_test_conversion_error(m);

    // type caster
    bind_test_tc_in_mat(m);
    bind_test_tc_in_row(m);
    bind_test_tc_in_col(m);
    bind_test_tc_in_cube(m);
    bind_test_tc_out_mat(m);
    bind_test_tc_out_mat_const(m);
    bind_test_tc_out_mat_rvalue(m);
    bind_test_tc_out_row(m);
    bind_test_tc_out_row_rvalue(m);
    bind_test_tc_out_col(m);
    bind_test_tc_out_col_rvalue(m);
    bind_test_tc_out_cube(m);
    bind_test_tc_out_cube_rvalue(m);

    // arraystore
    bind_ArrayStore<arma::Mat<int>>(m, std::string("i"));
    bind_ArrayStore<arma::Mat<long>>(m, std::string("l"));
    bind_ArrayStore<arma::Mat<float>>(m, std::string("f"));
    bind_ArrayStore<arma::Mat<double>>(m, std::string("d"));
    bind_test_ArrayStore_get_mat(m);
    bind_test_ArrayStore_get_mat_rvalue(m);
    bind_test_ArrayStore_get_view(m);

    // roundtrip
    bind_test_mat_roundtrip(m);
    bind_test_row_roundtrip(m);
    bind_test_col_roundtrip(m);
    bind_test_cube_roundtrip(m);
}
