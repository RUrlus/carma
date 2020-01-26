#include "test_mat_to_arr.h"

namespace carma { namespace tests {

py::array_t<double> test_mat_to_arr_return() {
    arma::mat mat = arma::randu<arma::Mat<double>>(4, 5);
    return mat_to_arr(mat, false);
} /* test_mat_to_arr */

int test_mat_to_arr(bool copy) {
    arma::Mat<double> M = arma::randu<arma::Mat<double>>(100, 2);

    double mat_sum = arma::accu(M);
    auto M_ptr = M.memptr();

    py::array_t<double> arr = mat_to_arr(M, copy);

    size_t arr_N = arr.size();
    size_t arr_S0 = arr.shape(0);
    size_t arr_S1 = arr.shape(1);
    auto arr_p = arr.unchecked<2>();

    // get buffer for raw pointer
    py::buffer_info info = arr.request();

    // compute sum of array
    double arr_sum = 0;
    for (size_t ci = 0; ci < arr_S1; ci++) {
    	for (size_t ri = 0; ri < arr_S0; ri++) {
			 arr_sum += arr_p(ri, ci);
		}
	}

    // variable for test status
    if (arr_N != 200) return 1;
    if (arr_S0 != 100) return 2;
    if (arr_S1 != 2) return 3;
    if (std::abs(arr_sum - mat_sum) > 1e-12) return 4;
    if (info.ptr != M_ptr) return 5;
    return 0;
} /* test_mat_to_arr */

int test_cube_to_arr(bool copy) {
    arma::Cube<double> M = arma::randu<arma::Cube<double>>(100, 2, 4);

    double mat_sum = arma::accu(M);
    auto M_ptr = M.memptr();

    py::array_t<double> arr = cube_to_arr(M, copy);

    size_t arr_N = arr.size();
    size_t arr_S0 = arr.shape(0);
    size_t arr_S1 = arr.shape(1);
    size_t arr_S2 = arr.shape(2);
    auto arr_p = arr.unchecked<3>();

    // get buffer for raw pointer
    py::buffer_info info = arr.request();

    // compute sum of array
    double arr_sum = 0;
    for (size_t si = 0; si < arr_S2; si++) {
        for (size_t ci = 0; ci < arr_S1; ci++) {
    	    for (size_t ri = 0; ri < arr_S0; ri++) {
			    arr_sum += arr_p(ri, ci, si);
		    }
	    }
    }
    // variable for test status
    if (arr_N != 800) return 1;
    if (arr_S0 != 4) return 2;
    if (arr_S1 != 100) return 3;
    if (arr_S2 != 2) return 3;
    if (std::abs(arr_sum - mat_sum) > 1e-12) return 4;
    if (info.ptr != M_ptr) return 5;
    return 0;
} /* test_cube_to_arr */

int test_row_to_arr(bool copy) {
    arma::Row<double> M = arma::randu<arma::Row<double>>(100);

    double mat_sum = arma::accu(M);
    auto M_ptr = M.memptr();

    py::array_t<double> arr = row_to_arr(M, copy);

    size_t arr_N = arr.size();
    size_t arr_S1 = arr.shape(1);
    auto arr_p = arr.unchecked<2>();

    // get buffer for raw pointer
    py::buffer_info info = arr.request();

    // compute sum of array
    double arr_sum = 0;
    for (size_t i = 0; i < arr_S1; i++) {
		arr_sum += arr_p(0, i);
    }

    // variable for test status
    if (arr_N != 100) return 1;
    if (arr_S1 != 100) return 2;
    if (std::abs(arr_sum - mat_sum) > 1e-12) return 4;
    if (info.ptr != M_ptr) return 5;
    return 0;
} /* test_row_to_arr */

int test_col_to_arr(bool copy) {
    arma::Col<double> M = arma::randu<arma::Col<double>>(100);

    double mat_sum = arma::accu(M);
    auto M_ptr = M.memptr();

    py::array_t<double> arr = col_to_arr(M, copy);

    size_t arr_N = arr.size();
    size_t arr_S0 = arr.shape(0);
    auto arr_p = arr.unchecked<2>();

    // get buffer for raw pointer
    py::buffer_info info = arr.request();

    // compute sum of array
    double arr_sum = 0;
    for (size_t i = 0; i < arr_S0; i++) {
		arr_sum += arr_p(i, 0);
    }

    // variable for test status
    if (arr_N != 100) return 1;
    if (arr_S0 != 100) return 2;
    if (std::abs(arr_sum - mat_sum) > 1e-12) return 4;
    if (info.ptr != M_ptr) return 5;
    return 0;
} /* test_row_to_arr */

} /* tests */ } /* carma */

void bind_test_mat_to_arr_return(py::module &m) {
    m.def(
        "mat_to_arr_return",
        &carma::tests::test_mat_to_arr_return,
        "Test mat_to_arr"
    );
}

void bind_test_mat_to_arr(py::module &m) {
    m.def(
        "mat_to_arr",
        &carma::tests::test_mat_to_arr,
        "Test mat_to_arr"
    );
}

void bind_test_cube_to_arr(py::module &m) {
    m.def(
        "cube_to_arr",
        &carma::tests::test_cube_to_arr,
        "Test cube_to_arr"
    );
}

void bind_test_row_to_arr(py::module &m) {
    m.def(
        "row_to_arr",
        &carma::tests::test_row_to_arr,
        "Test row_to_arr"
    );
}

void bind_test_col_to_arr(py::module &m) {
    m.def(
        "col_to_arr",
        &carma::tests::test_col_to_arr,
        "Test col_to_arr"
    );
}
