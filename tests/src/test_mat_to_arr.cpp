#include "test_mat_to_arr.h"

namespace carma { namespace tests {

// ------------------------------ Mat -----------------------------------------
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

py::array_t<double> test_mat_to_arr_return(bool copy) {
    return mat_to_arr<double>(arma::Mat<double>(4, 5, arma::fill::randu), copy);
} /* test_mat_to_arr_return */

py::array_t<double> test_mat_to_arr_plus_one(py::array_t<double> & arr, bool copy) {
    arma::Mat<double> ones = arma::ones(arr.shape(0), arr.shape(1));
    arma::Mat<double> mat = carma::arr_to_mat<double>(arr);
    arma::Mat<double> out = ones + mat;
    return mat_to_arr<double>(out, copy);
} /* test_mat_to_arr_plus_one */

// ------------------------------ Row -----------------------------------------

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

py::array_t<double> test_row_to_arr_return(bool copy) {
    return row_to_arr<double>(arma::Row<double>(100, arma::fill::randu), copy);
} /* test_row_to_arr_return */

py::array_t<double> test_row_to_arr_plus_one(py::array_t<double> & arr, bool copy) {
    arma::Row<double> ones = arma::Row<double>(arr.size(), arma::fill::ones);
    arma::Row<double> mat = carma::arr_to_row<double>(arr);
    arma::Row<double> out = ones + mat;
    return row_to_arr<double>(out, copy);
} /* test_row_to_arr_plus_one */

// ------------------------------ Col -----------------------------------------

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
} /* test_col_to_arr */

py::array_t<double> test_col_to_arr_return(bool copy) {
    return col_to_arr<double>(arma::Col<double>(100, arma::fill::randu), copy);
} /* test_col_to_arr_return */

py::array_t<double> test_col_to_arr_plus_one(py::array_t<double> & arr, bool copy) {
    arma::Col<double> ones = arma::ones(arr.size());
    arma::Col<double> mat = carma::arr_to_col<double>(arr);
    arma::Col<double> out = ones + mat;
    return col_to_arr<double>(out, copy);
} /* test_col_to_arr_plus_one */

// ------------------------------ Cube ----------------------------------------

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

py::array_t<double> test_cube_to_arr_return(bool copy) {
    return cube_to_arr<double>(arma::Cube<double>(100, 2, 4, arma::fill::randu), copy);
} /* test_cube_to_arr */

py::array_t<double> test_cube_to_arr_plus_one(py::array_t<double> & arr, bool copy) {
    arma::Cube<double> ones = arma::ones(arr.shape(0), arr.shape(1), arr.shape(2));
    arma::Cube<double> mat = carma::arr_to_cube<double>(arr);
    arma::Cube<double> out = ones + mat;
    return cube_to_arr<double>(out, copy);
} /* test_mat_to_arr_plus_one */


// ----------------------------------------------------------------------------
// ------------------------------ to_numpy ------------------------------------
// ----------------------------------------------------------------------------

int test_to_numpy_mat(bool copy) {
    arma::Mat<double> M = arma::randu<arma::Mat<double>>(100, 2);

    double mat_sum = arma::accu(M);
    auto M_ptr = M.memptr();

    py::array_t<double> arr = to_numpy<double>(M, copy);

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
} /* test_to_numpy_mat */

int test_to_numpy_cube(bool copy) {
    arma::Cube<double> M = arma::randu<arma::Cube<double>>(100, 2, 4);

    double mat_sum = arma::accu(M);
    auto M_ptr = M.memptr();

    py::array_t<double> arr = to_numpy<double>(M, copy);

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
} /* test_to_numpy_cube */

int test_to_numpy_row(bool copy) {
    arma::Row<double> M = arma::randu<arma::Row<double>>(100);

    double mat_sum = arma::accu(M);
    auto M_ptr = M.memptr();

    py::array_t<double> arr = to_numpy<double>(M, copy);

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
    if (arr_N != 100) return 1;
    if (arr_S0 != 1) return 2;
    if (arr_S1 != 100) return 3;
    if (std::abs(arr_sum - mat_sum) > 1e-12) return 4;
    if (info.ptr != M_ptr) return 5;
    return 0;
} /* test_to_numpy_row */

int test_to_numpy_col(bool copy) {
    arma::Col<double> M = arma::randu<arma::Col<double>>(100);

    double mat_sum = arma::accu(M);
    auto M_ptr = M.memptr();

    py::array_t<double> arr = to_numpy<double>(M, copy);

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
    if (arr_N != 100) return 1;
    if (arr_S0 != 100) return 2;
    if (arr_S1 != 1) return 3;
    if (std::abs(arr_sum - mat_sum) > 1e-12) return 4;
    if (info.ptr != M_ptr) return 5;
    return 0;
} /* test_to_numpy_col */

// ----------------------------------------------------------------------------
// ------------------------------ update_array --------------------------------
// ----------------------------------------------------------------------------

int test_update_array_mat(py::array_t<double> & arr, int cols) {

    arma::Mat<double> M = arr_to_mat<double>(arr, false, false);
    M.insert_cols(0, cols, true);

    update_array(M, arr);

    double mat_sum = arma::accu(M);
    auto M_ptr = M.memptr();

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
    if (arr_N != M.n_elem) return 1;
    if (arr_S0 != M.n_rows) return 2;
    if (arr_S1 != M.n_cols) return 3;
    if (std::abs(arr_sum - mat_sum) > 1e-12) return 4;
    if (info.ptr == M_ptr) return 5;
    return 0;
} /* test_update_array_mat */

int test_update_array_cube(py::array_t<double> & arr, int cols) {

    arma::Cube<double> M = arr_to_cube<double>(arr, false, false);
    M.insert_cols(0, cols, true);

    update_array(M, arr);

    double mat_sum = arma::accu(M);
    auto M_ptr = M.memptr();

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

    if (arr_N != M.n_elem) return 1;
    if (arr_S0 != M.n_rows) return 2;
    if (arr_S1 != M.n_cols) return 3;
    if (arr_S2 != M.n_slices) return 3;
    if (std::abs(arr_sum - mat_sum) > 1e-12) return 4;
    if (info.ptr == M_ptr) return 5;

    return 0;
} /* test_update_array_cube */

int test_update_array_row(py::array_t<double> & arr, int cols) {

    arma::Row<double> M = arr_to_row<double>(arr, false, false);
    M.insert_cols(0, cols, true);

    update_array(M, arr);

    double mat_sum = arma::accu(M);
    auto M_ptr = M.memptr();

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
    if (arr_N != M.n_elem) return 1;
    if (arr_S0 != M.n_rows) return 2;
    if (arr_S1 != M.n_cols) return 3;
    if (std::abs(arr_sum - mat_sum) > 1e-12) return 4;
    if (info.ptr == M_ptr) return 5;
    return 0;
} /* test_update_array_row */

int test_update_array_col(py::array_t<double> & arr, int cols) {

    arma::Col<double> M = arr_to_col<double>(arr, false, false);
    M.insert_rows(0, cols, true);

    update_array(M, arr);

    double mat_sum = arma::accu(M);
    auto M_ptr = M.memptr();

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
    if (arr_N != M.n_elem) return 1;
    if (arr_S0 != M.n_rows) return 2;
    if (arr_S1 != M.n_cols) return 3;
    if (std::abs(arr_sum - mat_sum) > 1e-12) return 4;
    if (info.ptr == M_ptr) return 5;
    return 0;
} /* test_update_array_col */

} /* tests */ } /* carma */

void bind_test_mat_to_arr_return(py::module &m) {
    m.def(
        "mat_to_arr_return",
        &carma::tests::test_mat_to_arr_return,
        "Test mat_to_arr"
    );
}

void bind_test_row_to_arr_return(py::module &m) {
    m.def(
        "row_to_arr_return",
        &carma::tests::test_row_to_arr_return,
        "Test row_to_arr"
    );
}

void bind_test_col_to_arr_return(py::module &m) {
    m.def(
        "col_to_arr_return",
        &carma::tests::test_col_to_arr_return,
        "Test col_to_arr"
    );
}

void bind_test_cube_to_arr_return(py::module &m) {
    m.def(
        "cube_to_arr_return",
        &carma::tests::test_cube_to_arr_return,
        "Test cube_to_arr"
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

void bind_test_to_numpy_mat(py::module &m) {
    m.def(
        "to_numpy_mat",
        &carma::tests::test_to_numpy_mat,
        "Test to_numpy"
    );
}

void bind_test_to_numpy_cube(py::module &m) {
    m.def(
        "to_numpy_cube",
        &carma::tests::test_to_numpy_cube,
        "Test to_numpy"
    );
}

void bind_test_to_numpy_row(py::module &m) {
    m.def(
        "to_numpy_row",
        &carma::tests::test_to_numpy_row,
        "Test to_numpy"
    );
}

void bind_test_to_numpy_col(py::module &m) {
    m.def(
        "to_numpy_col",
        &carma::tests::test_to_numpy_col,
        "Test to_numpy"
    );
}

void bind_test_update_array_mat(py::module &m) {
    m.def(
        "update_array_mat",
        &carma::tests::test_update_array_mat,
        "Test update_array"
    );
}

void bind_test_update_array_cube(py::module &m) {
    m.def(
        "update_array_cube",
        &carma::tests::test_update_array_cube,
        "Test update_array"
    );
}

void bind_test_update_array_row(py::module &m) {
    m.def(
        "update_array_row",
        &carma::tests::test_update_array_row,
        "Test update_array"
    );
}

void bind_test_update_array_col(py::module &m) {
    m.def(
        "update_array_col",
        &carma::tests::test_update_array_col,
        "Test update_array"
    );
}

void bind_test_mat_to_arr_plus_one(py::module &m) {
    m.def(
        "mat_to_arr_plus_one",
        &carma::tests::test_mat_to_arr_plus_one,
        "Test mat_to_arr"
    );
}

void bind_test_row_to_arr_plus_one(py::module &m) {
    m.def(
        "row_to_arr_plus_one",
        &carma::tests::test_row_to_arr_plus_one,
        "Test row_to_arr"
    );
}

void bind_test_col_to_arr_plus_one(py::module &m) {
    m.def(
        "col_to_arr_plus_one",
        &carma::tests::test_col_to_arr_plus_one,
        "Test col_to_arr"
    );
}

void bind_test_cube_to_arr_plus_one(py::module &m) {
    m.def(
        "cube_to_arr_plus_one",
        &carma::tests::test_cube_to_arr_plus_one,
        "Test cube_to_arr"
    );
}
