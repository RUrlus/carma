#include <pybind11/numpy.h>
#include <pybind11/pybind11.h>
#include <armadillo>
#include <carma>
#include <catch2/catch.hpp>

namespace py = pybind11;

typedef py::array_t<double, py::array::f_style | py::array::forcecast> fArr;

TEST_CASE("Test roundtrip mat", "[roundtrip_mat]") {
    py::module np = py::module::import("numpy");
    py::module np_rand = py::module::import("numpy.random");

    py::array_t<double> arr = fArr(np_rand.attr("normal")(0, 1, py::make_tuple(100, 2)));

    // attributes of the numpy array
    size_t arr_N = arr.size();
    size_t arr_S0 = arr.shape(0);
    size_t arr_S1 = arr.shape(1);

    // get buffer for raw pointer
    py::buffer_info info = arr.request();

    // compute sum of array
    double arr_sum = 0.0;
    auto _ptr = reinterpret_cast<double*>(info.ptr);
    auto ptr = arr.unchecked<2>();
    for (size_t ic = 0; ic < arr_S1; ic++) {
        for (size_t ir = 0; ir < arr_S0; ir++) {
            arr_sum += ptr(ir, ic);
        }
    }

    // call function to be tested
    arma::Mat<double> M = carma::arr_to_mat<double>(std::move(arr));

    double mat_sum = arma::accu(M);

    // variable for test status
    CHECK(arr_N == M.n_elem);
    CHECK(arr_S0 == M.n_rows);
    CHECK(arr_S1 == M.n_cols);
    CHECK(std::abs(arr_sum - mat_sum) < 1e-6);
    CHECK(_ptr == M.memptr());

    py::array_t<double> arr_out = carma::mat_to_arr(M);

    // attributes of the numpy array
    size_t arr_out_N = arr_out.size();
    size_t arr_out_S0 = arr_out.shape(0);
    size_t arr_out_S1 = arr_out.shape(1);

    // get buffer for raw pointer
    py::buffer_info info_out = arr_out.request();

    // compute sum of array
    double arr_sum_out = 0.0;
    auto _ptr_out = reinterpret_cast<double*>(info_out.ptr);
    auto ptr_out = arr_out.unchecked<2>();
    for (size_t ic = 0; ic < arr_out_S1; ic++) {
        for (size_t ir = 0; ir < arr_out_S0; ir++) {
            arr_sum_out += ptr_out(ir, ic);
        }
    }

    CHECK(arr_out_N == arr_N);
    CHECK(arr_out_S0 == arr_S0);
    CHECK(arr_out_S1 == arr_S1);
    CHECK(std::abs(arr_sum - arr_sum_out) < 1e-6);
    CHECK(_ptr == _ptr_out);
} /* TEST_CASE ROUNDTRIP_ROW */

TEST_CASE("Test roundtrip row", "[roundtrip_row]") {
    py::module np = py::module::import("numpy");
    py::module np_rand = py::module::import("numpy.random");

    py::array_t<double> arr = fArr(np_rand.attr("normal")(0, 1, 100));

    // attributes of the numpy array
    size_t arr_N = arr.size();
    size_t arr_S0 = arr.shape(0);

    // get buffer for raw pointer
    py::buffer_info info = arr.request();
    const double* ptr = reinterpret_cast<double*>(info.ptr);

    // compute sum of array
    double arr_sum = 0.0;
    for (size_t i = 0; i < static_cast<size_t>(arr.size()); i++)
        arr_sum += ptr[i];

    // // call function to be tested
    arma::Row<double> M = carma::arr_to_row<double>(std::move(arr));

    double mat_sum = arma::accu(M);

    // variable for test status
    CHECK(arr_N == M.n_elem);
    CHECK(arr_S0 == M.n_cols);
    CHECK(std::abs(arr_sum - mat_sum) < 1e-6);
    CHECK(info.ptr == M.memptr());

    py::array_t<double> arr_out = carma::row_to_arr(M);

    // attributes of the numpy array
    size_t arr_out_N = arr_out.size();
    size_t arr_out_S0 = arr_out.shape(1);

    // get buffer for raw pointer
    py::buffer_info info_out = arr_out.request();

    // compute sum of array
    double arr_sum_out = 0.0;
    auto ptr_out = arr_out.unchecked<2>();
    for (size_t i = 0; i < arr_out_S0; i++) {
        arr_sum_out += ptr_out(0, i);
    }

    CHECK(arr_out_N == arr_N);
    CHECK(arr_out_S0 == arr_S0);
    CHECK(std::abs(arr_sum - arr_sum_out) < 1e-6);
    CHECK(info.ptr == info_out.ptr);
} /* TEST_CASE ROUNDTRIP_ROW */

TEST_CASE("Test roundtrip col", "[roundtrip_col]") {
    py::module np = py::module::import("numpy");
    py::module np_rand = py::module::import("numpy.random");

    py::array_t<double> arr = fArr(np_rand.attr("normal")(0, 1, 100));

    // attributes of the numpy array
    size_t arr_N = arr.size();
    size_t arr_S0 = arr.shape(0);

    // get buffer for raw pointer
    py::buffer_info info = arr.request();
    const double* ptr = reinterpret_cast<double*>(info.ptr);

    // compute sum of array
    double arr_sum = 0.0;
    for (size_t i = 0; i < static_cast<size_t>(arr.size()); i++)
        arr_sum += ptr[i];

    // // call function to be tested
    arma::Col<double> M = carma::arr_to_col<double>(std::move(arr));

    double mat_sum = arma::accu(M);

    // variable for test status
    CHECK(arr_N == M.n_elem);
    CHECK(arr_S0 == M.n_rows);
    CHECK(std::abs(arr_sum - mat_sum) < 1e-6);
    CHECK(info.ptr == M.memptr());

    py::array_t<double> arr_out = carma::col_to_arr(M);

    // attributes of the numpy array
    size_t arr_out_N = arr_out.size();
    size_t arr_out_S0 = arr_out.shape(0);

    // get buffer for raw pointer
    py::buffer_info info_out = arr_out.request();

    // compute sum of array
    double arr_sum_out = 0.0;
    auto ptr_out = arr_out.unchecked<2>();
    for (size_t i = 0; i < arr_out_S0; i++) {
        arr_sum_out += ptr_out(i, 0);
    }

    CHECK(arr_out_N == arr_N);
    CHECK(arr_out_S0 == arr_S0);
    CHECK(std::abs(arr_sum - arr_sum_out) < 1e-6);
    CHECK(info.ptr == info_out.ptr);
} /* TEST_CASE ROUNDTRIP_COL */

TEST_CASE("Test roundtrip with cube", "[roundtrip_cube]") {
    py::module np = py::module::import("numpy");
    py::module np_rand = py::module::import("numpy.random");

    py::array_t<double> arr = fArr(np_rand.attr("normal")(0, 1, py::make_tuple(100, 4, 2)));

    // attributes of the numpy array
    size_t arr_N = arr.size();
    size_t arr_S0 = arr.shape(0);
    size_t arr_S1 = arr.shape(1);
    size_t arr_S2 = arr.shape(2);

    // get buffer for raw pointer
    py::buffer_info info = arr.request();

    // compute sum of array
    double arr_sum = 0.0;
    auto ptr = arr.unchecked<3>();
    for (size_t is = 0; is < arr_S2; is++) {
        for (size_t ic = 0; ic < arr_S1; ic++) {
            for (size_t ir = 0; ir < arr_S0; ir++) {
                arr_sum += ptr(ir, ic, is);
            }
        }
    }

    // call function to be tested
    arma::Cube<double> M = carma::arr_to_cube<double>(std::move(arr));

    double mat_sum = arma::accu(M);

    // variable for test status
    CHECK(arr_N == M.n_elem);
    CHECK(arr_S0 == M.n_rows);
    CHECK(arr_S1 == M.n_cols);
    CHECK(arr_S2 == M.n_slices);
    CHECK(std::abs(arr_sum - mat_sum) < 1e-6);
    CHECK(info.ptr == M.memptr());

    py::array_t<double> arr_out = carma::cube_to_arr(M);

    // attributes of the numpy array
    size_t arr_out_N = arr_out.size();
    size_t arr_out_S0 = arr_out.shape(0);
    size_t arr_out_S1 = arr_out.shape(1);
    size_t arr_out_S2 = arr_out.shape(2);

    // get buffer for raw pointer
    py::buffer_info info_out = arr_out.request();

    // compute sum of array
    double arr_out_sum = 0.0;
    auto ptr_out = arr_out.unchecked<3>();
    for (size_t is = 0; is < arr_out_S2; is++) {
        for (size_t ic = 0; ic < arr_out_S1; ic++) {
            for (size_t ir = 0; ir < arr_out_S0; ir++) {
                arr_out_sum += ptr_out(ir, ic, is);
            }
        }
    }

    // variable for test status
    CHECK(arr_out_N == arr_N);
    CHECK(arr_out_S0 == arr_S0);
    CHECK(arr_out_S1 == arr_S1);
    CHECK(arr_out_S2 == arr_S2);
    CHECK(std::abs(arr_sum - arr_out_sum) < 1e-6);
    CHECK(info_out.ptr == info.ptr);
} /* TEST_CASE ROUNDTRIP_CUBE */
