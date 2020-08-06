#include <carma/carma.h>
#include <pybind11/numpy.h>
#include <pybind11/pybind11.h>
#include <armadillo>
#include <catch2/catch.hpp>

namespace py = pybind11;

typedef py::array_t<double, py::array::f_style | py::array::forcecast> fArr;

TEST_CASE("Test arr_to_mat", "[arr_to_mat]") {
    py::module np = py::module::import("numpy");
    py::module np_rand = py::module::import("numpy.random");

    SECTION("F-contiguous; no copy; no strict") {
        bool copy = false;
        bool strict = false;

        py::array_t<double> arr = fArr(np_rand.attr("normal")(0, 1, py::make_tuple(100, 2)));

        // attributes of the numpy array
        size_t arr_N = arr.size();
        size_t arr_S0 = arr.shape(0);
        size_t arr_S1 = arr.shape(1);

        // get buffer for raw pointer
        py::buffer_info info = arr.request();
        const double* ptr = reinterpret_cast<double*>(info.ptr);

        // compute sum of array
        double arr_sum = 0.0;
        for (size_t i = 0; i < static_cast<size_t>(arr.size()); i++)
            arr_sum += ptr[i];

        // // call function to be tested
        arma::Mat<double> M = carma::arr_to_mat<double>(arr, copy, strict);

        double mat_sum = arma::accu(M);

        // variable for test status
        CHECK(arr_N == M.n_elem);
        CHECK(arr_S0 == M.n_rows);
        CHECK(arr_S1 == M.n_cols);
        CHECK(std::abs(arr_sum - mat_sum) < 1e-12);
        CHECK(info.ptr == M.memptr());
    }

    SECTION("C-contiguous; no copy; no strict") {
        bool copy = false;
        bool strict = false;

        py::array_t<double> arr = np_rand.attr("normal")(0, 1, py::make_tuple(100, 2));

        // attributes of the numpy array
        size_t arr_N = arr.size();
        size_t arr_S0 = arr.shape(0);
        size_t arr_S1 = arr.shape(1);

        // get buffer for raw pointer
        py::buffer_info info = arr.request();
        const double* ptr = reinterpret_cast<double*>(info.ptr);

        // compute sum of array
        double arr_sum = 0.0;
        for (size_t i = 0; i < static_cast<size_t>(arr.size()); i++)
            arr_sum += ptr[i];

        // // call function to be tested
        arma::Mat<double> M = carma::arr_to_mat<double>(arr, copy, strict);

        double mat_sum = arma::accu(M);

        // variable for test status
        CHECK(arr_N == M.n_elem);
        CHECK(arr_S0 == M.n_rows);
        CHECK(arr_S1 == M.n_cols);
        CHECK(std::abs(arr_sum - mat_sum) < 1e-12);
        CHECK(info.ptr != M.memptr());
    }

    SECTION("F-contiguous; copy; no strict") {
        bool copy = true;
        bool strict = false;

        py::array_t<double> arr = fArr(np_rand.attr("normal")(0, 1, py::make_tuple(100, 2)));

        // attributes of the numpy array
        size_t arr_N = arr.size();
        size_t arr_S0 = arr.shape(0);
        size_t arr_S1 = arr.shape(1);

        // get buffer for raw pointer
        py::buffer_info info = arr.request();
        const double* ptr = reinterpret_cast<double*>(info.ptr);

        // compute sum of array
        double arr_sum = 0.0;
        for (size_t i = 0; i < static_cast<size_t>(arr.size()); i++)
            arr_sum += ptr[i];

        // // call function to be tested
        arma::Mat<double> M = carma::arr_to_mat<double>(arr, copy, strict);

        double mat_sum = arma::accu(M);

        // variable for test status
        CHECK(arr_N == M.n_elem);
        CHECK(arr_S0 == M.n_rows);
        CHECK(arr_S1 == M.n_cols);
        CHECK(std::abs(arr_sum - mat_sum) < 1e-12);
        CHECK(info.ptr != M.memptr());
    }

    SECTION("C-contiguous; copy; no strict") {
        bool copy = true;
        bool strict = false;

        py::array_t<double> arr = np_rand.attr("normal")(0, 1, py::make_tuple(100, 2));

        // attributes of the numpy array
        size_t arr_N = arr.size();
        size_t arr_S0 = arr.shape(0);
        size_t arr_S1 = arr.shape(1);

        // get buffer for raw pointer
        py::buffer_info info = arr.request();
        const double* ptr = reinterpret_cast<double*>(info.ptr);

        // compute sum of array
        double arr_sum = 0.0;
        for (size_t i = 0; i < static_cast<size_t>(arr.size()); i++)
            arr_sum += ptr[i];

        // // call function to be tested
        arma::Mat<double> M = carma::arr_to_mat<double>(arr, copy, strict);

        double mat_sum = arma::accu(M);

        // variable for test status
        CHECK(arr_N == M.n_elem);
        CHECK(arr_S0 == M.n_rows);
        CHECK(arr_S1 == M.n_cols);
        CHECK(std::abs(arr_sum - mat_sum) < 1e-12);
        CHECK(info.ptr != M.memptr());
    }

    SECTION("F-contiguous; no copy; strict") {
        bool copy = false;
        bool strict = true;

        py::array_t<double> arr = fArr(np_rand.attr("normal")(0, 1, py::make_tuple(100, 2)));

        // attributes of the numpy array
        size_t arr_N = arr.size();
        size_t arr_S0 = arr.shape(0);
        size_t arr_S1 = arr.shape(1);

        // get buffer for raw pointer
        py::buffer_info info = arr.request();
        const double* ptr = reinterpret_cast<double*>(info.ptr);

        // compute sum of array
        double arr_sum = 0.0;
        for (size_t i = 0; i < static_cast<size_t>(arr.size()); i++)
            arr_sum += ptr[i];

        // // call function to be tested
        arma::Mat<double> M = carma::arr_to_mat<double>(arr, copy, strict);
        double mat_sum = arma::accu(M);

        // variable for test status
        CHECK(arr_N == M.n_elem);
        CHECK(arr_S0 == M.n_rows);
        CHECK((arr_S1) == M.n_cols);
        CHECK(std::abs(arr_sum - mat_sum) < 1e-12);
        CHECK(info.ptr == M.memptr());
    }

    SECTION("F-contiguous; no copy; no strict -- change") {
        bool copy = false;
        bool strict = false;

        py::array_t<double> arr = fArr(np_rand.attr("normal")(0, 1, py::make_tuple(100, 2)));

        // attributes of the numpy array
        size_t arr_N = arr.size();
        size_t arr_S0 = arr.shape(0);
        size_t arr_S1 = arr.shape(1);

        // get buffer for raw pointer
        py::buffer_info info = arr.request();
        const double* ptr = reinterpret_cast<double*>(info.ptr);

        // compute sum of array
        double arr_sum = 0.0;
        for (size_t i = 0; i < static_cast<size_t>(arr.size()); i++)
            arr_sum += ptr[i];

        // // call function to be tested
        arma::Mat<double> M = carma::arr_to_mat<double>(arr, copy, strict);

        M.insert_cols(0, 2, true);

        double mat_sum = arma::accu(M);

        // variable for test status
        CHECK((arr_N + 200) == M.n_elem);
        CHECK(arr_S0 == M.n_rows);
        CHECK((arr_S1 + 2) == M.n_cols);
        CHECK(std::abs(arr_sum - mat_sum) < 1e-12);
        CHECK(info.ptr != M.memptr());
    }

    SECTION("F-contiguous; no copy; strict -- change") {
        bool copy = false;
        bool strict = true;

        py::array_t<double> arr = fArr(np_rand.attr("normal")(0, 1, py::make_tuple(100, 2)));

        // // call function to be tested
        arma::Mat<double> M = carma::arr_to_mat<double>(arr, copy, strict);

        REQUIRE_THROWS(M.insert_cols(0, 2, true));
    }

    SECTION("dimension exception") {
        bool copy = false;
        bool strict = false;

        py::array_t<double> arr = fArr(np_rand.attr("normal")(0, 1, py::make_tuple(100, 2, 2)));

        REQUIRE_THROWS_AS(carma::arr_to_mat<double>(arr, copy, strict), carma::conversion_error);
    }
} /* TEST_CASE ARR_TO_MAT */

TEST_CASE("Test arr_to_row", "[arr_to_row]") {
    py::module np = py::module::import("numpy");
    py::module np_rand = py::module::import("numpy.random");

    SECTION("F-contiguous; no copy; no strict") {
        bool copy = false;
        bool strict = false;

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
        arma::Row<double> M = carma::arr_to_row<double>(arr, copy, strict);

        double mat_sum = arma::accu(M);

        // variable for test status
        CHECK(arr_N == M.n_elem);
        CHECK(arr_S0 == M.n_cols);
        CHECK(std::abs(arr_sum - mat_sum) < 1e-12);
        CHECK(info.ptr == M.memptr());
    }

    SECTION("2D; no copy; no strict") {
        bool copy = false;
        bool strict = false;

        py::array_t<double> arr = fArr(np_rand.attr("normal")(0, 1, py::make_tuple(1, 100)));

        // attributes of the numpy array
        size_t arr_N = arr.size();
        size_t arr_S0 = arr.shape(0);
        size_t arr_S1 = arr.shape(1);

        // get buffer for raw pointer
        py::buffer_info info = arr.request();
        const double* ptr = reinterpret_cast<double*>(info.ptr);

        // compute sum of array
        double arr_sum = 0.0;
        for (size_t i = 0; i < static_cast<size_t>(arr.size()); i++)
            arr_sum += ptr[i];

        // // call function to be tested
        arma::Row<double> M = carma::arr_to_row<double>(arr, copy, strict);

        double mat_sum = arma::accu(M);

        // variable for test status
        CHECK(arr_N == M.n_elem);
        CHECK(arr_S0 == M.n_rows);
        CHECK(arr_S1 == M.n_cols);
        CHECK(std::abs(arr_sum - mat_sum) < 1e-12);
        CHECK(info.ptr == M.memptr());
    }

    SECTION("C-contiguous 2D; no copy; no strict") {
        bool copy = false;
        bool strict = false;

        py::array_t<double> arr = np_rand.attr("normal")(0, 1, py::make_tuple(1, 100));

        // attributes of the numpy array
        size_t arr_N = arr.size();
        size_t arr_S0 = arr.shape(0);
        size_t arr_S1 = arr.shape(1);

        // get buffer for raw pointer
        py::buffer_info info = arr.request();
        const double* ptr = reinterpret_cast<double*>(info.ptr);

        // compute sum of array
        double arr_sum = 0.0;
        for (size_t i = 0; i < static_cast<size_t>(arr.size()); i++)
            arr_sum += ptr[i];

        // // call function to be tested
        arma::Row<double> M = carma::arr_to_row<double>(arr, copy, strict);

        double mat_sum = arma::accu(M);

        // variable for test status
        CHECK(arr_N == M.n_elem);
        CHECK(arr_S0 == M.n_rows);
        CHECK(arr_S1 == M.n_cols);
        CHECK(std::abs(arr_sum - mat_sum) < 1e-12);
        CHECK(info.ptr == M.memptr());
    }

    SECTION("C-contiguous; no copy; no strict") {
        bool copy = false;
        bool strict = false;

        py::array_t<double> arr = np_rand.attr("normal")(0, 1, 100);

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
        arma::Row<double> M = carma::arr_to_row<double>(arr, copy, strict);

        double mat_sum = arma::accu(M);

        // variable for test status
        CHECK(arr_N == M.n_elem);
        CHECK(arr_S0 == M.n_cols);
        CHECK(std::abs(arr_sum - mat_sum) < 1e-12);
        CHECK(info.ptr == M.memptr());
    }

    SECTION("copy; no strict") {
        bool copy = true;
        bool strict = false;

        py::array_t<double> arr = np_rand.attr("normal")(0, 1, 100);

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
        arma::Row<double> M = carma::arr_to_row<double>(arr, copy, strict);

        double mat_sum = arma::accu(M);

        // variable for test status
        CHECK(arr_N == M.n_elem);
        CHECK(arr_S0 == M.n_cols);
        CHECK(std::abs(arr_sum - mat_sum) < 1e-12);
        CHECK(info.ptr != M.memptr());
    }

    SECTION("no copy; strict") {
        bool copy = false;
        bool strict = true;

        py::array_t<double> arr = np_rand.attr("normal")(0, 1, 100);

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
        arma::Row<double> M = carma::arr_to_row<double>(arr, copy, strict);
        double mat_sum = arma::accu(M);

        // variable for test status
        CHECK(arr_N == M.n_elem);
        CHECK(arr_S0 == M.n_cols);
        CHECK(std::abs(arr_sum - mat_sum) < 1e-12);
        CHECK(info.ptr == M.memptr());
    }

    SECTION("no copy; no strict -- change") {
        bool copy = false;
        bool strict = false;

        py::array_t<double> arr = np_rand.attr("normal")(0, 1, 100);

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
        arma::Row<double> M = carma::arr_to_row<double>(arr, copy, strict);

        M.insert_cols(0, 2, true);

        double mat_sum = arma::accu(M);

        // variable for test status
        CHECK(arr_N + 2 == M.n_elem);
        CHECK(arr_S0 + 2 == M.n_cols);
        CHECK(std::abs(arr_sum - mat_sum) < 1e-12);
        CHECK(info.ptr != M.memptr());
    }

    SECTION("no copy; strict -- change") {
        bool copy = false;
        bool strict = true;

        py::array_t<double> arr = np_rand.attr("normal")(0, 1, 100);

        // // call function to be tested
        arma::Row<double> M = carma::arr_to_row<double>(arr, copy, strict);

        REQUIRE_THROWS(M.insert_cols(0, 2, true));
    }

    SECTION("dimension exception") {
        bool copy = false;
        bool strict = false;

        py::array_t<double> arr = fArr(np_rand.attr("normal")(0, 1, py::make_tuple(2, 100)));

        REQUIRE_THROWS_AS(carma::arr_to_row<double>(arr, copy, strict), carma::conversion_error);
    }
} /* TEST_CASE ARR_TO_ROW */

TEST_CASE("Test arr_to_col", "[arr_to_col]") {
    py::module np = py::module::import("numpy");
    py::module np_rand = py::module::import("numpy.random");

    SECTION("F-contiguous; no copy; no strict") {
        bool copy = false;
        bool strict = false;

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
        arma::Col<double> M = carma::arr_to_col<double>(arr, copy, strict);

        double mat_sum = arma::accu(M);

        // variable for test status
        CHECK(arr_N == M.n_elem);
        CHECK(arr_S0 == M.n_rows);
        CHECK(std::abs(arr_sum - mat_sum) < 1e-12);
        CHECK(info.ptr == M.memptr());
    }

    SECTION("2D; no copy; no strict") {
        bool copy = false;
        bool strict = false;

        py::array_t<double> arr = fArr(np_rand.attr("normal")(0, 1, py::make_tuple(100, 1)));

        // attributes of the numpy array
        size_t arr_N = arr.size();
        size_t arr_S0 = arr.shape(0);
        size_t arr_S1 = arr.shape(1);

        // get buffer for raw pointer
        py::buffer_info info = arr.request();
        const double* ptr = reinterpret_cast<double*>(info.ptr);

        // compute sum of array
        double arr_sum = 0.0;
        for (size_t i = 0; i < static_cast<size_t>(arr.size()); i++)
            arr_sum += ptr[i];

        // // call function to be tested
        arma::Col<double> M = carma::arr_to_col<double>(arr, copy, strict);

        double mat_sum = arma::accu(M);

        // variable for test status
        CHECK(arr_N == M.n_elem);
        CHECK(arr_S0 == M.n_rows);
        CHECK(arr_S1 == M.n_cols);
        CHECK(std::abs(arr_sum - mat_sum) < 1e-12);
        CHECK(info.ptr == M.memptr());
    }

    SECTION("C-contiguous 2D; no copy; no strict") {
        bool copy = false;
        bool strict = false;

        py::array_t<double> arr = np_rand.attr("normal")(0, 1, py::make_tuple(100, 1));

        // attributes of the numpy array
        size_t arr_N = arr.size();
        size_t arr_S0 = arr.shape(0);
        size_t arr_S1 = arr.shape(1);

        // get buffer for raw pointer
        py::buffer_info info = arr.request();
        const double* ptr = reinterpret_cast<double*>(info.ptr);

        // compute sum of array
        double arr_sum = 0.0;
        for (size_t i = 0; i < static_cast<size_t>(arr.size()); i++)
            arr_sum += ptr[i];

        // // call function to be tested
        arma::Col<double> M = carma::arr_to_col<double>(arr, copy, strict);

        double mat_sum = arma::accu(M);

        // variable for test status
        CHECK(arr_N == M.n_elem);
        CHECK(arr_S0 == M.n_rows);
        CHECK(arr_S1 == M.n_cols);
        CHECK(std::abs(arr_sum - mat_sum) < 1e-12);
        CHECK(info.ptr == M.memptr());
    }

    SECTION("C-contiguous; no copy; no strict") {
        bool copy = false;
        bool strict = false;

        py::array_t<double> arr = np_rand.attr("normal")(0, 1, 100);

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
        arma::Col<double> M = carma::arr_to_col<double>(arr, copy, strict);

        double mat_sum = arma::accu(M);

        // variable for test status
        CHECK(arr_N == M.n_elem);
        CHECK(arr_S0 == M.n_rows);
        CHECK(std::abs(arr_sum - mat_sum) < 1e-12);
        CHECK(info.ptr == M.memptr());
    }

    SECTION("copy; no strict") {
        bool copy = true;
        bool strict = false;

        py::array_t<double> arr = np_rand.attr("normal")(0, 1, 100);

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
        arma::Col<double> M = carma::arr_to_col<double>(arr, copy, strict);

        double mat_sum = arma::accu(M);

        // variable for test status
        CHECK(arr_N == M.n_elem);
        CHECK(arr_S0 == M.n_rows);
        CHECK(std::abs(arr_sum - mat_sum) < 1e-12);
        CHECK(info.ptr != M.memptr());
    }

    SECTION("no copy; strict") {
        bool copy = false;
        bool strict = true;

        py::array_t<double> arr = np_rand.attr("normal")(0, 1, 100);

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
        arma::Col<double> M = carma::arr_to_col<double>(arr, copy, strict);
        double mat_sum = arma::accu(M);

        // variable for test status
        CHECK(arr_N == M.n_elem);
        CHECK(arr_S0 == M.n_rows);
        CHECK(std::abs(arr_sum - mat_sum) < 1e-12);
        CHECK(info.ptr == M.memptr());
    }

    SECTION("no copy; no strict -- change") {
        bool copy = false;
        bool strict = false;

        py::array_t<double> arr = np_rand.attr("normal")(0, 1, 100);

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
        arma::Col<double> M = carma::arr_to_col<double>(arr, copy, strict);

        M.insert_rows(0, 2, true);

        double mat_sum = arma::accu(M);

        // variable for test status
        CHECK(arr_N + 2 == M.n_elem);
        CHECK(arr_S0 + 2 == M.n_rows);
        CHECK(std::abs(arr_sum - mat_sum) < 1e-12);
        CHECK(info.ptr != M.memptr());
    }

    SECTION("no copy; strict -- change") {
        bool copy = false;
        bool strict = true;

        py::array_t<double> arr = np_rand.attr("normal")(0, 1, 100);

        // // call function to be tested
        arma::Col<double> M = carma::arr_to_col<double>(arr, copy, strict);

        REQUIRE_THROWS(M.insert_cols(0, 2, true));
    }

    SECTION("dimension exception") {
        bool copy = false;
        bool strict = false;

        py::array_t<double> arr = fArr(np_rand.attr("normal")(0, 1, py::make_tuple(100, 2)));

        REQUIRE_THROWS_AS(carma::arr_to_col<double>(arr, copy, strict), carma::conversion_error);
    }
} /* TEST_CASE ARR_TO_COL */

TEST_CASE("Test arr_to_cube", "[arr_to_cube]") {
    py::module np = py::module::import("numpy");
    py::module np_rand = py::module::import("numpy.random");

    SECTION("F-contiguous; no copy; no strict") {
        bool copy = false;
        bool strict = false;

        py::array_t<double> arr = fArr(np_rand.attr("normal")(0, 1, py::make_tuple(100, 2, 2)));

        // attributes of the numpy array
        size_t arr_N = arr.size();
        size_t arr_S0 = arr.shape(0);
        size_t arr_S1 = arr.shape(1);
        size_t arr_S2 = arr.shape(2);

        // get buffer for raw pointer
        py::buffer_info info = arr.request();
        const double* ptr = reinterpret_cast<double*>(info.ptr);

        // compute sum of array
        double arr_sum = 0.0;
        for (size_t i = 0; i < static_cast<size_t>(arr.size()); i++)
            arr_sum += ptr[i];

        // // call function to be tested
        arma::Cube<double> M = carma::arr_to_cube<double>(arr, copy, strict);

        double mat_sum = arma::accu(M);

        // variable for test status
        CHECK(arr_N == M.n_elem);
        CHECK(arr_S0 == M.n_rows);
        CHECK(arr_S1 == M.n_cols);
        CHECK(arr_S2 == M.n_slices);
        CHECK(std::abs(arr_sum - mat_sum) < 1e-12);
        CHECK(info.ptr == M.memptr());
    }

    SECTION("C-contiguous; no copy; no strict") {
        bool copy = false;
        bool strict = false;

        py::array_t<double> arr = np_rand.attr("normal")(0, 1, py::make_tuple(100, 2, 2));

        // attributes of the numpy array
        size_t arr_N = arr.size();
        size_t arr_S0 = arr.shape(0);
        size_t arr_S1 = arr.shape(1);
        size_t arr_S2 = arr.shape(2);

        // get buffer for raw pointer
        py::buffer_info info = arr.request();
        const double* ptr = reinterpret_cast<double*>(info.ptr);

        // compute sum of array
        double arr_sum = 0.0;
        for (size_t i = 0; i < static_cast<size_t>(arr.size()); i++)
            arr_sum += ptr[i];

        // // call function to be tested
        arma::Cube<double> M = carma::arr_to_cube<double>(arr, copy, strict);

        double mat_sum = arma::accu(M);

        // variable for test status
        CHECK(arr_N == M.n_elem);
        CHECK(arr_S0 == M.n_rows);
        CHECK(arr_S1 == M.n_cols);
        CHECK(arr_S2 == M.n_slices);
        CHECK(std::abs(arr_sum - mat_sum) < 1e-12);
        CHECK(info.ptr != M.memptr());
    }

    SECTION("F-contiguous; copy; no strict") {
        bool copy = true;
        bool strict = false;

        py::array_t<double> arr = fArr(np_rand.attr("normal")(0, 1, py::make_tuple(100, 2, 2)));

        // attributes of the numpy array
        size_t arr_N = arr.size();
        size_t arr_S0 = arr.shape(0);
        size_t arr_S1 = arr.shape(1);
        size_t arr_S2 = arr.shape(2);

        // get buffer for raw pointer
        py::buffer_info info = arr.request();
        const double* ptr = reinterpret_cast<double*>(info.ptr);

        // compute sum of array
        double arr_sum = 0.0;
        for (size_t i = 0; i < static_cast<size_t>(arr.size()); i++)
            arr_sum += ptr[i];

        // // call function to be tested
        arma::Cube<double> M = carma::arr_to_cube<double>(arr, copy, strict);

        double mat_sum = arma::accu(M);

        // variable for test status
        CHECK(arr_N == M.n_elem);
        CHECK(arr_S0 == M.n_rows);
        CHECK(arr_S1 == M.n_cols);
        CHECK(arr_S2 == M.n_slices);
        CHECK(std::abs(arr_sum - mat_sum) < 1e-12);
        CHECK(info.ptr != M.memptr());
    }

    SECTION("C-contiguous; copy; no strict") {
        bool copy = true;
        bool strict = false;

        py::array_t<double> arr = np_rand.attr("normal")(0, 1, py::make_tuple(100, 2, 2));

        // attributes of the numpy array
        size_t arr_N = arr.size();
        size_t arr_S0 = arr.shape(0);
        size_t arr_S1 = arr.shape(1);
        size_t arr_S2 = arr.shape(2);

        // get buffer for raw pointer
        py::buffer_info info = arr.request();
        const double* ptr = reinterpret_cast<double*>(info.ptr);

        // compute sum of array
        double arr_sum = 0.0;
        for (size_t i = 0; i < static_cast<size_t>(arr.size()); i++)
            arr_sum += ptr[i];

        // // call function to be tested
        arma::Cube<double> M = carma::arr_to_cube<double>(arr, copy, strict);

        double mat_sum = arma::accu(M);

        // variable for test status
        CHECK(arr_N == M.n_elem);
        CHECK(arr_S0 == M.n_rows);
        CHECK(arr_S1 == M.n_cols);
        CHECK(arr_S2 == M.n_slices);
        CHECK(std::abs(arr_sum - mat_sum) < 1e-12);
        CHECK(info.ptr != M.memptr());
    }

    SECTION("F-contiguous; no copy; strict") {
        bool copy = false;
        bool strict = true;

        py::array_t<double> arr = fArr(np_rand.attr("normal")(0, 1, py::make_tuple(100, 2, 2)));

        // attributes of the numpy array
        size_t arr_N = arr.size();
        size_t arr_S0 = arr.shape(0);
        size_t arr_S1 = arr.shape(1);
        size_t arr_S2 = arr.shape(2);

        // get buffer for raw pointer
        py::buffer_info info = arr.request();
        const double* ptr = reinterpret_cast<double*>(info.ptr);

        // compute sum of array
        double arr_sum = 0.0;
        for (size_t i = 0; i < static_cast<size_t>(arr.size()); i++)
            arr_sum += ptr[i];

        // // call function to be tested
        arma::Cube<double> M = carma::arr_to_cube<double>(arr, copy, strict);
        double mat_sum = arma::accu(M);

        // variable for test status
        CHECK(arr_N == M.n_elem);
        CHECK(arr_S0 == M.n_rows);
        CHECK((arr_S1) == M.n_cols);
        CHECK((arr_S2) == M.n_slices);
        CHECK(std::abs(arr_sum - mat_sum) < 1e-12);
        CHECK(info.ptr == M.memptr());
    }

    SECTION("F-contiguous; no copy; no strict -- change") {
        bool copy = false;
        bool strict = false;

        py::array_t<double> arr = fArr(np_rand.attr("normal")(0, 1, py::make_tuple(100, 2, 2)));

        // attributes of the numpy array
        size_t arr_S0 = arr.shape(0);
        size_t arr_S1 = arr.shape(1);

        // get buffer for raw pointer
        py::buffer_info info = arr.request();
        const double* ptr = reinterpret_cast<double*>(info.ptr);

        // compute sum of array
        double arr_sum = 0.0;
        for (size_t i = 0; i < static_cast<size_t>(arr.size()); i++)
            arr_sum += ptr[i];

        // // call function to be tested
        arma::Cube<double> M = carma::arr_to_cube<double>(arr, copy, strict);

        M.insert_cols(0, 2, true);

        double mat_sum = arma::accu(M);

        // variable for test status
        CHECK(arr_S0 == M.n_rows);
        CHECK((arr_S1 + 2) == M.n_cols);
        CHECK((arr_S1) == M.n_slices);
        CHECK(std::abs(arr_sum - mat_sum) < 1e-12);
        CHECK(info.ptr != M.memptr());
    }

    SECTION("F-contiguous; no copy; strict -- change") {
        bool copy = false;
        bool strict = true;

        py::array_t<double> arr = fArr(np_rand.attr("normal")(0, 1, py::make_tuple(100, 2, 2)));

        // // call function to be tested
        arma::Cube<double> M = carma::arr_to_cube<double>(arr, copy, strict);

        REQUIRE_THROWS(M.insert_cols(0, 2, true));
    }

    SECTION("dimension exception") {
        bool copy = false;
        bool strict = false;

        py::array_t<double> arr = fArr(np_rand.attr("normal")(0, 1, py::make_tuple(100, 2)));

        REQUIRE_THROWS_AS(carma::arr_to_cube<double>(arr, copy, strict), carma::conversion_error);
    }
} /* TEST_CASE ARR_TO_CUBE */
