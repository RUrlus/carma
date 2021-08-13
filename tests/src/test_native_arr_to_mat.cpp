#include <pybind11/numpy.h>
#include <pybind11/pybind11.h>
#include <armadillo>
#include <carma>
#include <catch2/catch.hpp>

namespace py = pybind11;

typedef py::array_t<double, py::array::f_style | py::array::forcecast> fArr;

TEST_CASE("Version check", "[version_check]") {
    std::cout << "carma v" << carma::carma_version().as_string() << "\n";
}

TEST_CASE("Test arr_to_mat", "[arr_to_mat]") {
    py::module np = py::module::import("numpy");
    py::module np_rand = py::module::import("numpy.random");

    SECTION("F-contiguous; no copy") {
        int copy = 0;

        py::array_t<double> arr = fArr(np_rand.attr("normal")(0, 1, py::make_tuple(100, 2)));

        // attributes of the numpy array
        size_t arr_N = arr.size();
        size_t arr_S0 = arr.shape(0);
        size_t arr_S1 = arr.shape(1);

        // get buffer for raw pointer
        py::buffer_info info = arr.request();

        // compute sum of array
        double arr_sum = 0.0;
        auto* _ptr = reinterpret_cast<double*>(info.ptr);
        auto ptr = arr.unchecked<2>();
        for (size_t ic = 0; ic < arr_S1; ic++) {
            for (size_t ir = 0; ir < arr_S0; ir++) {
                arr_sum += ptr(ir, ic);
            }
        }

        // call function to be tested
        arma::Mat<double> M = carma::arr_to_mat<double>(arr, copy);

        double mat_sum = arma::accu(M);

        // variable for test status
        CHECK(arr_N == M.n_elem);
        CHECK(arr_S0 == M.n_rows);
        CHECK(arr_S1 == M.n_cols);
        CHECK(std::abs(arr_sum - mat_sum) < 1e-6);
        CHECK(_ptr == M.memptr());
    }

    SECTION("F-contiguous; copy") {
        bool copy = true;

        py::array_t<double> arr = fArr(np_rand.attr("normal")(0, 1, py::make_tuple(100, 2)));

        // attributes of the numpy array
        size_t arr_N = arr.size();
        size_t arr_S0 = arr.shape(0);
        size_t arr_S1 = arr.shape(1);

        // get buffer for raw pointer
        py::buffer_info info = arr.request();

        // compute sum of array
        double arr_sum = 0.0;
        auto* _ptr = reinterpret_cast<double*>(info.ptr);
        auto ptr = arr.unchecked<2>();
        for (size_t ic = 0; ic < arr_S1; ic++) {
            for (size_t ir = 0; ir < arr_S0; ir++) {
                arr_sum += ptr(ir, ic);
            }
        }

        // call function to be tested
        arma::Mat<double> M = carma::arr_to_mat<double>(arr, copy);

        double mat_sum = arma::accu(M);

        // variable for test status
        CHECK(arr_N == M.n_elem);
        CHECK(arr_S0 == M.n_rows);
        CHECK(arr_S1 == M.n_cols);
        CHECK(std::abs(arr_sum - mat_sum) < 1e-6);
        CHECK(_ptr != M.memptr());
    }

    SECTION("F-contiguous; steal") {
        py::array_t<double> arr = fArr(np_rand.attr("normal")(0, 1, py::make_tuple(100, 2)));

        // attributes of the numpy array
        size_t arr_N = arr.size();
        size_t arr_S0 = arr.shape(0);
        size_t arr_S1 = arr.shape(1);

        // get buffer for raw pointer
        py::buffer_info info = arr.request();

        // compute sum of array
        double arr_sum = 0.0;
        auto* _ptr = reinterpret_cast<double*>(info.ptr);
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
    }

    SECTION("F-contiguous; const") {
        const py::array_t<double> arr = fArr(np_rand.attr("normal")(0, 1, py::make_tuple(100, 2)));

        // attributes of the numpy array
        size_t arr_N = arr.size();
        size_t arr_S0 = arr.shape(0);
        size_t arr_S1 = arr.shape(1);

        // get buffer for raw pointer
        py::buffer_info info = arr.request();

        // compute sum of array
        double arr_sum = 0.0;
        auto* _ptr = reinterpret_cast<double*>(info.ptr);
        auto ptr = arr.unchecked<2>();
        for (size_t ic = 0; ic < arr_S1; ic++) {
            for (size_t ir = 0; ir < arr_S0; ir++) {
                arr_sum += ptr(ir, ic);
            }
        }

        // call function to be tested
        arma::Mat<double> M = carma::arr_to_mat<double>(arr);

        double mat_sum = arma::accu(M);

        // variable for test status
        CHECK(arr_N == M.n_elem);
        CHECK(arr_S0 == M.n_rows);
        CHECK(arr_S1 == M.n_cols);
        CHECK(std::abs(arr_sum - mat_sum) < 1e-6);
        CHECK(_ptr != M.memptr());
    }

    SECTION("C-contiguous; no copy") {
        int copy = 0;

        py::array_t<double> arr = np_rand.attr("normal")(0, 1, py::make_tuple(100, 2));

        // attributes of the numpy array
        size_t arr_N = arr.size();
        size_t arr_S0 = arr.shape(0);
        size_t arr_S1 = arr.shape(1);
        auto arr_p = arr.unchecked<2>();

        // get buffer for raw pointer
        py::buffer_info info = arr.request();
        auto ptr = reinterpret_cast<double*>(info.ptr);

        // compute sum of array
        double arr_sum = 0;
        for (size_t ci = 0; ci < arr_S1; ci++) {
            for (size_t ri = 0; ri < arr_S0; ri++) {
                arr_sum += arr_p(ri, ci);
            }
        }

        // call function to be tested
        arma::Mat<double> M = carma::arr_to_mat<double>(arr, copy);

        double mat_sum = arma::accu(M);

        // variable for test status
        CHECK(arr_N == M.n_elem);
        CHECK(arr_S0 == M.n_rows);
        CHECK(arr_S1 == M.n_cols);
        INFO("is c-contiguous " << carma::is_c_contiguous(arr));
        INFO("is f-contiguous " << carma::is_f_contiguous(arr));
        INFO("is aligned " << carma::is_aligned(arr));
        INFO("mat_sum is  " << mat_sum);
        INFO("arr_sum is  " << arr_sum);
        INFO("M " << M);
        CHECK(std::abs(arr_sum - mat_sum) < 1e-6);
        CHECK(ptr != M.memptr());
    }

    SECTION("C-contiguous; copy") {
        bool copy = true;

        py::array_t<double> arr = np_rand.attr("normal")(0, 1, py::make_tuple(100, 2));

        // attributes of the numpy array
        size_t arr_N = arr.size();
        size_t arr_S0 = arr.shape(0);
        size_t arr_S1 = arr.shape(1);
        auto arr_p = arr.unchecked<2>();

        // get buffer for raw pointer
        py::buffer_info info = arr.request();
        auto* ptr = reinterpret_cast<double*>(info.ptr);

        // compute sum of array
        double arr_sum = 0;
        for (size_t ci = 0; ci < arr_S1; ci++) {
            for (size_t ri = 0; ri < arr_S0; ri++) {
                arr_sum += arr_p(ri, ci);
            }
        }

        // call function to be tested
        arma::Mat<double> M = carma::arr_to_mat<double>(arr, copy);

        double mat_sum = arma::accu(M);

        // variable for test status
        CHECK(arr_N == M.n_elem);
        CHECK(arr_S0 == M.n_rows);
        CHECK(arr_S1 == M.n_cols);
        INFO("is c-contiguous " << carma::is_c_contiguous(arr));
        INFO("is f-contiguous " << carma::is_f_contiguous(arr));
        INFO("is aligned " << carma::is_aligned(arr));
        INFO("mat_sum is  " << mat_sum);
        INFO("arr_sum is  " << arr_sum);
        INFO("M " << M);
        CHECK(std::abs(arr_sum - mat_sum) < 1e-6);
        CHECK(ptr != M.memptr());
    }

    SECTION("C-contiguous; steal") {
        py::array_t<double> arr = np_rand.attr("normal")(0, 1, py::make_tuple(100, 2));

        // attributes of the numpy array
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

        // call function to be tested
        arma::Mat<double> M = carma::arr_to_mat<double>(std::move(arr));

        double mat_sum = arma::accu(M);

        // variable for test status
        CHECK(arr_N == M.n_elem);
        CHECK(arr_S0 == M.n_rows);
        CHECK(arr_S1 == M.n_cols);
        INFO("is c-contiguous " << carma::is_c_contiguous(arr));
        INFO("is f-contiguous " << carma::is_f_contiguous(arr));
        INFO("is aligned " << carma::is_aligned(arr));
        INFO("mat_sum is  " << mat_sum);
        INFO("arr_sum is  " << arr_sum);
        INFO("M " << M);
        CHECK(std::abs(arr_sum - mat_sum) < 1e-6);
        CHECK(info.ptr != M.memptr());
    }

    SECTION("C-contiguous; const") {
        const py::array_t<double> arr = np_rand.attr("normal")(0, 1, py::make_tuple(100, 2));

        // attributes of the numpy array
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

        // call function to be tested
        arma::Mat<double> M = carma::arr_to_mat<double>(std::move(arr));

        double mat_sum = arma::accu(M);

        // variable for test status
        CHECK(arr_N == M.n_elem);
        CHECK(arr_S0 == M.n_rows);
        CHECK(arr_S1 == M.n_cols);
        INFO("is c-contiguous " << carma::is_c_contiguous(arr));
        INFO("is f-contiguous " << carma::is_f_contiguous(arr));
        INFO("is aligned " << carma::is_aligned(arr));
        INFO("mat_sum is  " << mat_sum);
        INFO("arr_sum is  " << arr_sum);
        INFO("M " << M);
        CHECK(std::abs(arr_sum - mat_sum) < 1e-6);
        CHECK(info.ptr != M.memptr());
        M.insert_cols(0, 2, true);
    }

    SECTION("F-contiguous; no copy; change") {
        int copy = 0;

        py::array_t<double> arr = fArr(np_rand.attr("normal")(0, 1, py::make_tuple(100, 2)));

        // call function to be tested
        arma::Mat<double> M = carma::arr_to_mat<double>(arr, copy);

        REQUIRE_THROWS(M.insert_cols(0, 2, true));
    }

    SECTION("F-contiguous; copy; change") {
        int copy = 1;

        py::array_t<double> arr = fArr(np_rand.attr("normal")(0, 1, py::make_tuple(100, 2)));

        // call function to be tested
        arma::Mat<double> M = carma::arr_to_mat<double>(arr, copy);
        M.insert_cols(0, 2, true);
    }

    SECTION("F-contiguous; const; change") {
        const py::array_t<double> arr = fArr(np_rand.attr("normal")(0, 1, py::make_tuple(100, 2)));

        // call function to be tested
        arma::Mat<double> M = carma::arr_to_mat<double>(arr);
        M.insert_cols(0, 2, true);
    }

    SECTION("F-contiguous; steal; change") {
        py::array_t<double> arr = fArr(np_rand.attr("normal")(0, 1, py::make_tuple(100, 2)));

        // call function to be tested
        arma::Mat<double> M = carma::arr_to_mat<double>(std::move(arr));
        M.insert_cols(0, 2, true);
    }

    SECTION("C-contiguous; no copy; change") {
        py::array_t<double> arr = np_rand.attr("normal")(0, 1, py::make_tuple(100, 2));

        // call function to be tested
        arma::Mat<double> M = carma::arr_to_mat<double>(arr, false);

        REQUIRE_THROWS(M.insert_cols(0, 2, true));
    }

    SECTION("dimension exception") {
        py::array_t<double> arr = fArr(np_rand.attr("normal")(0, 1, py::make_tuple(100, 2, 2)));

        REQUIRE_THROWS_AS(carma::arr_to_mat<double>(arr, false), carma::ConversionError);
    }
} /* TEST_CASE ARR_TO_MAT */

TEST_CASE("Test arr_to_row", "[arr_to_row]") {
    py::module np = py::module::import("numpy");
    py::module np_rand = py::module::import("numpy.random");

    SECTION("F-contiguous; no copy") {
        bool copy = false;

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

        //  call function to be tested
        arma::Row<double> M = carma::arr_to_row<double>(arr, copy);

        double mat_sum = arma::accu(M);

        // variable for test status
        CHECK(arr_N == M.n_elem);
        CHECK(arr_S0 == M.n_cols);
        CHECK(std::abs(arr_sum - mat_sum) < 1e-6);
        CHECK(info.ptr == M.memptr());
    }

    SECTION("F-contiguous; copy") {
        bool copy = true;

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

        //  call function to be tested
        arma::Row<double> M = carma::arr_to_row<double>(arr, copy);

        double mat_sum = arma::accu(M);

        // variable for test status
        CHECK(arr_N == M.n_elem);
        CHECK(arr_S0 == M.n_cols);
        CHECK(std::abs(arr_sum - mat_sum) < 1e-6);
        CHECK(info.ptr != M.memptr());
    }

    SECTION("F-contiguous; steal") {
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

        //  call function to be tested
        arma::Row<double> M = carma::arr_to_row<double>(std::move(arr));

        double mat_sum = arma::accu(M);

        // variable for test status
        CHECK(arr_N == M.n_elem);
        CHECK(arr_S0 == M.n_cols);
        CHECK(std::abs(arr_sum - mat_sum) < 1e-6);
        CHECK(info.ptr == M.memptr());
    }

    SECTION("F-contiguous; const") {
        const py::array_t<double> arr = fArr(np_rand.attr("normal")(0, 1, 100));

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

        //  call function to be tested
        arma::Row<double> M = carma::arr_to_row<double>(arr);

        double mat_sum = arma::accu(M);

        // variable for test status
        CHECK(arr_N == M.n_elem);
        CHECK(arr_S0 == M.n_cols);
        CHECK(std::abs(arr_sum - mat_sum) < 1e-6);
        CHECK(info.ptr != M.memptr());
    }

    SECTION("2D; no copy") {
        bool copy = false;

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

        //  call function to be tested
        arma::Row<double> M = carma::arr_to_row<double>(arr, copy);

        double mat_sum = arma::accu(M);

        // variable for test status
        CHECK(arr_N == M.n_elem);
        CHECK(arr_S0 == M.n_rows);
        CHECK(arr_S1 == M.n_cols);
        CHECK(std::abs(arr_sum - mat_sum) < 1e-6);
        CHECK(info.ptr == M.memptr());
    }

    SECTION("C-contiguous 2D; no copy") {
        bool copy = false;

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

        //  call function to be tested
        arma::Row<double> M = carma::arr_to_row<double>(arr, copy);

        double mat_sum = arma::accu(M);

        // variable for test status
        CHECK(arr_N == M.n_elem);
        CHECK(arr_S0 == M.n_rows);
        CHECK(arr_S1 == M.n_cols);
        CHECK(std::abs(arr_sum - mat_sum) < 1e-6);
        CHECK(info.ptr == M.memptr());
    }

    SECTION("C-contiguous; no copy") {
        bool copy = false;

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

        //  call function to be tested
        arma::Row<double> M = carma::arr_to_row<double>(arr, copy);

        double mat_sum = arma::accu(M);

        // variable for test status
        CHECK(arr_N == M.n_elem);
        CHECK(arr_S0 == M.n_cols);
        CHECK(std::abs(arr_sum - mat_sum) < 1e-6);
        CHECK(info.ptr == M.memptr());
    }

    SECTION("C-contiguous; copy") {
        bool copy = true;

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

        //  call function to be tested
        arma::Row<double> M = carma::arr_to_row<double>(arr, copy);

        double mat_sum = arma::accu(M);

        // variable for test status
        CHECK(arr_N == M.n_elem);
        CHECK(arr_S0 == M.n_cols);
        CHECK(std::abs(arr_sum - mat_sum) < 1e-6);
        CHECK(info.ptr != M.memptr());
    }

    SECTION("C-contiguous; steal") {
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

        //  call function to be tested
        arma::Row<double> M = carma::arr_to_row<double>(std::move(arr));

        double mat_sum = arma::accu(M);

        // variable for test status
        CHECK(arr_N == M.n_elem);
        CHECK(arr_S0 == M.n_cols);
        CHECK(std::abs(arr_sum - mat_sum) < 1e-6);
        CHECK(info.ptr == M.memptr());
    }

    SECTION("C-contiguous; const") {
        const py::array_t<double> arr = np_rand.attr("normal")(0, 1, 100);

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

        //  call function to be tested
        arma::Row<double> M = carma::arr_to_row<double>(arr);

        double mat_sum = arma::accu(M);

        // variable for test status
        CHECK(arr_N == M.n_elem);
        CHECK(arr_S0 == M.n_cols);
        CHECK(std::abs(arr_sum - mat_sum) < 1e-6);
        CHECK(info.ptr != M.memptr());
    }

    SECTION("no copy; change") {
        bool copy = false;

        py::array_t<double> arr = np_rand.attr("normal")(0, 1, 100);

        //  call function to be tested
        arma::Row<double> M = carma::arr_to_row<double>(arr, copy);
        REQUIRE_THROWS(M.insert_cols(0, 2, true));
    }

    SECTION("copy; change") {
        bool copy = true;

        py::array_t<double> arr = np_rand.attr("normal")(0, 1, 100);

        //  call function to be tested
        arma::Row<double> M = carma::arr_to_row<double>(arr, copy);
        M.insert_cols(0, 2, true);
    }

    SECTION("const; change") {
        const py::array_t<double> arr = np_rand.attr("normal")(0, 1, 100);

        //  call function to be tested
        arma::Row<double> M = carma::arr_to_row<double>(arr);
        M.insert_cols(0, 2, true);
    }

    SECTION("steal; change") {
        py::array_t<double> arr = np_rand.attr("normal")(0, 1, 100);

        //  call function to be tested
        arma::Row<double> M = carma::arr_to_row<double>(std::move(arr));
        M.insert_cols(0, 2, true);
    }

    SECTION("dimension exception") {
        bool copy = false;

        py::array_t<double> arr = fArr(np_rand.attr("normal")(0, 1, py::make_tuple(2, 100)));

        REQUIRE_THROWS_AS(carma::arr_to_row<double>(arr, copy), carma::ConversionError);
    }
} /* TEST_CASE ARR_TO_ROW */

TEST_CASE("Test arr_to_col", "[arr_to_col]") {
    py::module np = py::module::import("numpy");
    py::module np_rand = py::module::import("numpy.random");

    SECTION("F-contiguous; no copy") {
        bool copy = false;

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

        //  call function to be tested
        arma::Col<double> M = carma::arr_to_col<double>(arr, copy);

        double mat_sum = arma::accu(M);

        // variable for test status
        CHECK(arr_N == M.n_elem);
        CHECK(arr_S0 == M.n_rows);
        CHECK(std::abs(arr_sum - mat_sum) < 1e-6);
        CHECK(info.ptr == M.memptr());
    }

    SECTION("F-contiguous; copy") {
        bool copy = true;

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

        //  call function to be tested
        arma::Col<double> M = carma::arr_to_col<double>(arr, copy);

        double mat_sum = arma::accu(M);

        // variable for test status
        CHECK(arr_N == M.n_elem);
        CHECK(arr_S0 == M.n_rows);
        CHECK(std::abs(arr_sum - mat_sum) < 1e-6);
        CHECK(info.ptr != M.memptr());
    }

    SECTION("F-contiguous; steal") {
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

        //  call function to be tested
        arma::Col<double> M = carma::arr_to_col<double>(std::move(arr));

        double mat_sum = arma::accu(M);

        // variable for test status
        CHECK(arr_N == M.n_elem);
        CHECK(arr_S0 == M.n_rows);
        CHECK(std::abs(arr_sum - mat_sum) < 1e-6);
        CHECK(info.ptr == M.memptr());
    }

    SECTION("F-contiguous; const") {
        const py::array_t<double> arr = fArr(np_rand.attr("normal")(0, 1, 100));

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

        //  call function to be tested
        arma::Col<double> M = carma::arr_to_col<double>(arr);

        double mat_sum = arma::accu(M);

        // variable for test status
        CHECK(arr_N == M.n_elem);
        CHECK(arr_S0 == M.n_rows);
        CHECK(std::abs(arr_sum - mat_sum) < 1e-6);
        CHECK(info.ptr != M.memptr());
    }

    SECTION("2D; no copy") {
        bool copy = false;

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

        //  call function to be tested
        arma::Col<double> M = carma::arr_to_col<double>(arr, copy);

        double mat_sum = arma::accu(M);

        // variable for test status
        CHECK(arr_N == M.n_elem);
        CHECK(arr_S0 == M.n_rows);
        CHECK(arr_S1 == M.n_cols);
        CHECK(std::abs(arr_sum - mat_sum) < 1e-6);
        CHECK(info.ptr == M.memptr());
    }

    SECTION("C-contiguous 2D; no copy") {
        bool copy = false;

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

        //  call function to be tested
        arma::Col<double> M = carma::arr_to_col<double>(arr, copy);

        double mat_sum = arma::accu(M);

        // variable for test status
        CHECK(arr_N == M.n_elem);
        CHECK(arr_S0 == M.n_rows);
        CHECK(arr_S1 == M.n_cols);
        CHECK(std::abs(arr_sum - mat_sum) < 1e-6);
        CHECK(info.ptr == M.memptr());
    }

    SECTION("C-contiguous; no copy") {
        bool copy = false;

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

        //  call function to be tested
        arma::Col<double> M = carma::arr_to_col<double>(arr, copy);

        double mat_sum = arma::accu(M);

        // variable for test status
        CHECK(arr_N == M.n_elem);
        CHECK(arr_S0 == M.n_rows);
        CHECK(std::abs(arr_sum - mat_sum) < 1e-6);
        CHECK(info.ptr == M.memptr());
    }

    SECTION("C-contiguous; copy") {
        bool copy = true;

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

        //  call function to be tested
        arma::Col<double> M = carma::arr_to_col<double>(arr, copy);

        double mat_sum = arma::accu(M);

        // variable for test status
        CHECK(arr_N == M.n_elem);
        CHECK(arr_S0 == M.n_rows);
        CHECK(std::abs(arr_sum - mat_sum) < 1e-6);
        CHECK(info.ptr != M.memptr());
    }

    SECTION("C-contiguous; steal") {
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

        //  call function to be tested
        arma::Col<double> M = carma::arr_to_col<double>(std::move(arr));

        double mat_sum = arma::accu(M);

        // variable for test status
        CHECK(arr_N == M.n_elem);
        CHECK(arr_S0 == M.n_rows);
        CHECK(std::abs(arr_sum - mat_sum) < 1e-6);
        CHECK(info.ptr == M.memptr());
    }

    SECTION("C-contiguous; const") {
        const py::array_t<double> arr = np_rand.attr("normal")(0, 1, 100);

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

        //  call function to be tested
        arma::Col<double> M = carma::arr_to_col<double>(arr);

        double mat_sum = arma::accu(M);

        // variable for test status
        CHECK(arr_N == M.n_elem);
        CHECK(arr_S0 == M.n_rows);
        CHECK(std::abs(arr_sum - mat_sum) < 1e-6);
        CHECK(info.ptr != M.memptr());
    }

    SECTION("no copy; change") {
        bool copy = false;

        py::array_t<double> arr = np_rand.attr("normal")(0, 1, 100);

        // call function to be tested
        arma::Col<double> M = carma::arr_to_col<double>(arr, copy);

        REQUIRE_THROWS(M.insert_rows(0, 2, true));
    }

    SECTION("copy; change") {
        bool copy = true;

        py::array_t<double> arr = np_rand.attr("normal")(0, 1, 100);

        // call function to be tested
        arma::Col<double> M = carma::arr_to_col<double>(arr, copy);
        M.insert_rows(0, 2, true);
    }

    SECTION("steal; change") {
        py::array_t<double> arr = np_rand.attr("normal")(0, 1, 100);

        //  call function to be tested
        arma::Col<double> M = carma::arr_to_col<double>(std::move(arr));
        M.insert_rows(0, 2, true);
    }

    SECTION("const; change") {
        const py::array_t<double> arr = np_rand.attr("normal")(0, 1, 100);

        //  call function to be tested
        arma::Col<double> M = carma::arr_to_col<double>(arr);
        M.insert_rows(0, 2, true);
    }

    SECTION("dimension exception") {
        bool copy = false;

        py::array_t<double> arr = fArr(np_rand.attr("normal")(0, 1, py::make_tuple(100, 2)));

        REQUIRE_THROWS_AS(carma::arr_to_col<double>(arr, copy), carma::ConversionError);
    }
} /* TEST_CASE ARR_TO_COL */

TEST_CASE("Test arr_to_cube", "[arr_to_cube]") {
    py::module np = py::module::import("numpy");
    py::module np_rand = py::module::import("numpy.random");

    SECTION("F-contiguous; no copy") {
        bool copy = false;

        py::array_t<double> arr = fArr(np_rand.attr("normal")(0, 1, py::make_tuple(100, 2, 2)));

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
        arma::Cube<double> M = carma::arr_to_cube<double>(arr, copy);

        double mat_sum = arma::accu(M);

        // variable for test status
        CHECK(arr_N == M.n_elem);
        CHECK(arr_S0 == M.n_rows);
        CHECK(arr_S1 == M.n_cols);
        CHECK(arr_S2 == M.n_slices);
        CHECK(std::abs(arr_sum - mat_sum) < 1e-6);
        CHECK(info.ptr == M.memptr());
    }

    SECTION("F-contiguous; copy") {
        bool copy = true;

        py::array_t<double> arr = fArr(np_rand.attr("normal")(0, 1, py::make_tuple(100, 2, 2)));

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
        arma::Cube<double> M = carma::arr_to_cube<double>(arr, copy);

        double mat_sum = arma::accu(M);

        // variable for test status
        CHECK(arr_N == M.n_elem);
        CHECK(arr_S0 == M.n_rows);
        CHECK(arr_S1 == M.n_cols);
        CHECK(arr_S2 == M.n_slices);
        CHECK(std::abs(arr_sum - mat_sum) < 1e-6);
        CHECK(info.ptr != M.memptr());
    }

    SECTION("F-contiguous; steal") {
        py::array_t<double> arr = fArr(np_rand.attr("normal")(0, 1, py::make_tuple(100, 2, 2)));

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
    }

    SECTION("F-contiguous; const") {
        const py::array_t<double> arr = fArr(np_rand.attr("normal")(0, 1, py::make_tuple(100, 2, 2)));

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
        arma::Cube<double> M = carma::arr_to_cube<double>(arr);

        double mat_sum = arma::accu(M);

        // variable for test status
        CHECK(arr_N == M.n_elem);
        CHECK(arr_S0 == M.n_rows);
        CHECK(arr_S1 == M.n_cols);
        CHECK(arr_S2 == M.n_slices);
        CHECK(std::abs(arr_sum - mat_sum) < 1e-6);
        CHECK(info.ptr != M.memptr());
    }

    SECTION("C-contiguous; no copy") {
        bool copy = false;

        py::array_t<double> arr = np_rand.attr("normal")(0, 1, py::make_tuple(100, 2, 2));

        // attributes of the numpy array
        size_t arr_N = arr.size();
        size_t arr_S0 = arr.shape(0);
        size_t arr_S1 = arr.shape(1);
        size_t arr_S2 = arr.shape(2);
        auto arr_p = arr.unchecked<3>();

        // get buffer for raw pointer
        py::buffer_info info = arr.request();
        auto ptr = reinterpret_cast<double*>(info.ptr);

        // compute sum of array
        double arr_sum = 0;
        for (size_t si = 0; si < arr_S2; si++) {
            for (size_t ci = 0; ci < arr_S1; ci++) {
                for (size_t ri = 0; ri < arr_S0; ri++) {
                    arr_sum += arr_p(ri, ci, si);
                }
            }
        }

        // call function to be tested
        arma::Cube<double> M = carma::arr_to_cube<double>(arr, copy);

        double mat_sum = arma::accu(M);

        // variable for test status
        CHECK(arr_N == M.n_elem);
        CHECK(arr_S0 == M.n_rows);
        CHECK(arr_S1 == M.n_cols);
        CHECK(arr_S2 == M.n_slices);
        INFO("is c-contiguous " << carma::is_c_contiguous(arr));
        INFO("is f-contiguous " << carma::is_f_contiguous(arr));
        INFO("is aligned " << carma::is_aligned(arr));
        INFO("mat_sum is  " << mat_sum);
        INFO("arr_sum is  " << arr_sum);
        INFO("M " << M);
        CHECK(std::abs(arr_sum - mat_sum) < 1e-6);
        CHECK(ptr != M.memptr());
    }

    SECTION("C-contiguous; copy") {
        bool copy = true;

        py::array_t<double> arr = np_rand.attr("normal")(0, 1, py::make_tuple(100, 2, 2));

        // attributes of the numpy array
        size_t arr_N = arr.size();
        size_t arr_S0 = arr.shape(0);
        size_t arr_S1 = arr.shape(1);
        size_t arr_S2 = arr.shape(2);
        auto arr_p = arr.unchecked<3>();

        // get buffer for raw pointer
        py::buffer_info info = arr.request();
        double* ptr = reinterpret_cast<double*>(info.ptr);

        // compute sum of array
        double arr_sum = 0;
        for (size_t si = 0; si < arr_S2; si++) {
            for (size_t ci = 0; ci < arr_S1; ci++) {
                for (size_t ri = 0; ri < arr_S0; ri++) {
                    arr_sum += arr_p(ri, ci, si);
                }
            }
        }

        // call function to be tested
        arma::Cube<double> M = carma::arr_to_cube<double>(arr, copy);

        double mat_sum = arma::accu(M);

        // variable for test status
        CHECK(arr_N == M.n_elem);
        CHECK(arr_S0 == M.n_rows);
        CHECK(arr_S1 == M.n_cols);
        CHECK(arr_S2 == M.n_slices);
        INFO("is c-contiguous " << carma::is_c_contiguous(arr));
        INFO("is f-contiguous " << carma::is_f_contiguous(arr));
        INFO("is aligned " << carma::is_aligned(arr));
        INFO("mat_sum is  " << mat_sum);
        INFO("arr_sum is  " << arr_sum);
        INFO("M " << M);
        CHECK(std::abs(arr_sum - mat_sum) < 1e-6);
        CHECK(ptr != M.memptr());
    }

    SECTION("C-contiguous; steal") {
        py::array_t<double> arr = np_rand.attr("normal")(0, 1, py::make_tuple(100, 2, 2));

        // attributes of the numpy array
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

        // call function to be tested
        arma::Cube<double> M = carma::arr_to_cube<double>(std::move(arr));

        double mat_sum = arma::accu(M);

        // variable for test status
        CHECK(arr_N == M.n_elem);
        CHECK(arr_S0 == M.n_rows);
        CHECK(arr_S1 == M.n_cols);
        CHECK(arr_S2 == M.n_slices);
        INFO("is c-contiguous " << carma::is_c_contiguous(arr));
        INFO("is f-contiguous " << carma::is_f_contiguous(arr));
        INFO("is aligned " << carma::is_aligned(arr));
        INFO("mat_sum is  " << mat_sum);
        INFO("arr_sum is  " << arr_sum);
        INFO("M " << M);
        CHECK(std::abs(arr_sum - mat_sum) < 1e-6);
        CHECK(info.ptr != M.memptr());
    }

    SECTION("C-contiguous; const") {
        const py::array_t<double> arr = np_rand.attr("normal")(0, 1, py::make_tuple(100, 2, 2));

        // attributes of the numpy array
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

        // call function to be tested
        arma::Cube<double> M = carma::arr_to_cube<double>(arr);

        double mat_sum = arma::accu(M);

        // variable for test status
        CHECK(arr_N == M.n_elem);
        CHECK(arr_S0 == M.n_rows);
        CHECK(arr_S1 == M.n_cols);
        CHECK(arr_S2 == M.n_slices);
        INFO("is c-contiguous " << carma::is_c_contiguous(arr));
        INFO("is f-contiguous " << carma::is_f_contiguous(arr));
        INFO("is aligned " << carma::is_aligned(arr));
        INFO("mat_sum is  " << mat_sum);
        INFO("arr_sum is  " << arr_sum);
        INFO("M " << M);
        CHECK(std::abs(arr_sum - mat_sum) < 1e-6);
        CHECK(info.ptr != M.memptr());
    }

    SECTION("F-contiguous; steal; -- change") {
        py::array_t<double> arr = fArr(np_rand.attr("normal")(0, 1, py::make_tuple(100, 2, 2)));

        // attributes of the numpy array
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

        M.insert_cols(0, 2, true);

        double mat_sum = arma::accu(M);

        // variable for test status
        CHECK(arr_S0 == M.n_rows);
        CHECK((arr_S1 + 2) == M.n_cols);
        CHECK((arr_S1) == M.n_slices);
        CHECK(std::abs(arr_sum - mat_sum) < 1e-6);
        CHECK(info.ptr != M.memptr());
    }

#ifndef WIN32
    SECTION("F-contiguous; no copy; -- change") {
        bool copy = false;

        py::array_t<double> arr = fArr(np_rand.attr("normal")(0, 1, py::make_tuple(100, 2, 2)));

        // call function to be tested
        arma::Cube<double> M = carma::arr_to_cube<double>(arr, copy);

        REQUIRE_THROWS(M.insert_slices(0, 2, true));
    }
#endif

#ifndef WIN32
    SECTION("F-contiguous; const; -- change") {
        const py::array_t<double> arr = fArr(np_rand.attr("normal")(0, 1, py::make_tuple(100, 2, 2)));

        // call function to be tested
        arma::Cube<double> M = carma::arr_to_cube<double>(arr);
        M.insert_cols(0, 2, true);
        M.insert_slices(0, 2, true);
    }
#endif

#ifndef WIN32
    SECTION("F-contiguous; steal; -- change") {
        py::array_t<double> arr = fArr(np_rand.attr("normal")(0, 1, py::make_tuple(100, 2, 2)));

        // call function to be tested
        arma::Cube<double> M = carma::arr_to_cube<double>(std::move(arr));
        M.insert_cols(0, 2, true);
        M.insert_slices(0, 2, true);
    }
#endif

    SECTION("dimension exception") {
        bool copy = false;

        py::array_t<double> arr = fArr(np_rand.attr("normal")(0, 1, py::make_tuple(100, 2)));

        REQUIRE_THROWS_AS(carma::arr_to_cube<double>(arr, copy), carma::ConversionError);
    }
} /* TEST_CASE ARR_TO_CUBE */

TEST_CASE("Test arr_to_mat_view", "[arr_to_mat_view]") {
    py::module np = py::module::import("numpy");
    py::module np_rand = py::module::import("numpy.random");

    SECTION("F-contiguous") {
        const py::array_t<double> arr = fArr(np_rand.attr("normal")(0, 1, py::make_tuple(100, 2)));

        // attributes of the numpy array
        size_t arr_N = arr.size();
        size_t arr_S0 = arr.shape(0);
        size_t arr_S1 = arr.shape(1);

        // get buffer for raw pointer
        py::buffer_info info = arr.request();

        // compute sum of array
        double arr_sum = 0.0;
        auto* _ptr = reinterpret_cast<double*>(info.ptr);
        auto ptr = arr.unchecked<2>();
        for (size_t ic = 0; ic < arr_S1; ic++) {
            for (size_t ir = 0; ir < arr_S0; ir++) {
                arr_sum += ptr(ir, ic);
            }
        }

        // call function to be tested
        const arma::Mat<double> M = carma::arr_to_mat_view<double>(arr);

        double mat_sum = arma::accu(M);

        // variable for test status
        CHECK(arr_N == M.n_elem);
        CHECK(arr_S0 == M.n_rows);
        CHECK(arr_S1 == M.n_cols);
        CHECK(std::abs(arr_sum - mat_sum) < 1e-6);
        CHECK(_ptr == M.memptr());
    }

    SECTION("C-contiguous") {
        const py::array_t<double> arr = np_rand.attr("normal")(0, 1, py::make_tuple(100, 2));

        // attributes of the numpy array
        size_t arr_N = arr.size();
        size_t arr_S0 = arr.shape(0);
        size_t arr_S1 = arr.shape(1);

        // get buffer for raw pointer
        py::buffer_info info = arr.request();

        // compute sum of array
        double arr_sum = 0.0;
        double* _ptr = reinterpret_cast<double*>(info.ptr);
        auto ptr = arr.unchecked<2>();
        for (size_t ic = 0; ic < arr_S1; ic++) {
            for (size_t ir = 0; ir < arr_S0; ir++) {
                arr_sum += ptr(ir, ic);
            }
        }

        // call function to be tested
        const arma::Mat<double> M = carma::arr_to_mat_view<double>(arr);

        double mat_sum = arma::accu(M);

        // variable for test status
        CHECK(arr_N == M.n_elem);
        CHECK(arr_S0 == M.n_rows);
        CHECK(arr_S1 == M.n_cols);
        CHECK(std::abs(arr_sum - mat_sum) < 1e-6);
        CHECK(_ptr != M.memptr());
    }
}  // TEST_CASE ARR_TO_MAT_VIEW

TEST_CASE("Test arr_to_row_view", "[arr_to_row_view]") {
    py::module np = py::module::import("numpy");
    py::module np_rand = py::module::import("numpy.random");

    SECTION("F-contiguous") {
        const py::array_t<double> arr = fArr(np_rand.attr("normal")(0, 1, 100));

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

        //  call function to be tested
        const arma::Row<double> M = carma::arr_to_row_view<double>(arr);

        double mat_sum = arma::accu(M);

        // variable for test status
        CHECK(arr_N == M.n_elem);
        CHECK(arr_S0 == M.n_cols);
        CHECK(std::abs(arr_sum - mat_sum) < 1e-6);
        CHECK(info.ptr == M.memptr());
    }

    SECTION("C-contiguous") {
        const py::array_t<double> arr = np_rand.attr("normal")(0, 1, 100);

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

        //  call function to be tested
        const arma::Row<double> M = carma::arr_to_row_view<double>(arr);

        double mat_sum = arma::accu(M);

        // variable for test status
        CHECK(arr_N == M.n_elem);
        CHECK(arr_S0 == M.n_cols);
        CHECK(std::abs(arr_sum - mat_sum) < 1e-6);
        CHECK(info.ptr == M.memptr());
    }
}  // TEST_CASE ARR_TO_ROW_VIEW

TEST_CASE("Test arr_to_col_view", "[arr_to_col_view]") {
    py::module np = py::module::import("numpy");
    py::module np_rand = py::module::import("numpy.random");

    SECTION("F-contiguous") {
        const py::array_t<double> arr = fArr(np_rand.attr("normal")(0, 1, 100));

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

        //  call function to be tested
        const arma::Col<double> M = carma::arr_to_col_view<double>(arr);

        double mat_sum = arma::accu(M);

        // variable for test status
        CHECK(arr_N == M.n_elem);
        CHECK(arr_S0 == M.n_rows);
        CHECK(std::abs(arr_sum - mat_sum) < 1e-6);
        CHECK(info.ptr == M.memptr());
    }

    SECTION("C-contiguous") {
        const py::array_t<double> arr = np_rand.attr("normal")(0, 1, 100);

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

        //  call function to be tested
        const arma::Col<double> M = carma::arr_to_col_view<double>(arr);

        double mat_sum = arma::accu(M);

        // variable for test status
        CHECK(arr_N == M.n_elem);
        CHECK(arr_S0 == M.n_rows);
        CHECK(std::abs(arr_sum - mat_sum) < 1e-6);
        CHECK(info.ptr == M.memptr());
    }
}  // TEST_CASE ARR_TO_COL_VIEW

TEST_CASE("Test arr_to_cube_view", "[arr_to_cube_view]") {
    py::module np = py::module::import("numpy");
    py::module np_rand = py::module::import("numpy.random");

    SECTION("F-contiguous") {
        const py::array_t<double> arr = fArr(np_rand.attr("normal")(0, 1, py::make_tuple(100, 2, 2)));

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
        const arma::Cube<double> M = carma::arr_to_cube_view<double>(arr);

        double mat_sum = arma::accu(M);

        // variable for test status
        CHECK(arr_N == M.n_elem);
        CHECK(arr_S0 == M.n_rows);
        CHECK(arr_S1 == M.n_cols);
        CHECK(arr_S2 == M.n_slices);
        CHECK(std::abs(arr_sum - mat_sum) < 1e-6);
        CHECK(info.ptr == M.memptr());
    }

    SECTION("C-contiguous") {
        const py::array_t<double> arr = np_rand.attr("normal")(0, 1, py::make_tuple(100, 2, 2));

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
        const arma::Cube<double> M = carma::arr_to_cube_view<double>(arr);

        double mat_sum = arma::accu(M);

        // variable for test status
        CHECK(arr_N == M.n_elem);
        CHECK(arr_S0 == M.n_rows);
        CHECK(arr_S1 == M.n_cols);
        CHECK(arr_S2 == M.n_slices);
        CHECK(std::abs(arr_sum - mat_sum) < 1e-6);
        CHECK(info.ptr != M.memptr());
    }
}  // TEST_CASE ARR_TO_CUBE_VIEW
