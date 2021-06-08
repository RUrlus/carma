#include <pybind11/numpy.h>
#include <pybind11/pybind11.h>
#include <armadillo>
#include <carma>

namespace py = pybind11;

namespace carma {
namespace tests {

py::array_t<double> test_mat_roundtrip(py::array_t<double>& arr) {
    // call function to be tested
    arma::Mat<double> M = carma::arr_to_mat<double>(std::move(arr));
    return carma::mat_to_arr(M);
}

py::array_t<double> test_row_roundtrip(py::array_t<double>& arr) {
    // call function to be tested
    arma::Row<double> M = carma::arr_to_row<double>(std::move(arr));
    return carma::row_to_arr(M);
}

py::array_t<double> test_col_roundtrip(py::array_t<double>& arr) {
    // call function to be tested
    arma::Col<double> M = carma::arr_to_col<double>(std::move(arr));
    return carma::col_to_arr(M);
}

py::array_t<double> test_cube_roundtrip(py::array_t<double>& arr) {
    // call function to be tested
    arma::Cube<double> M = carma::arr_to_cube<double>(std::move(arr));
    return carma::cube_to_arr(M);
}

}  // namespace tests
}  // namespace carma

void bind_test_mat_roundtrip(py::module& m) {
    m.def("mat_roundtrip", &carma::tests::test_mat_roundtrip, "Test mat_roundtrip");
}

void bind_test_row_roundtrip(py::module& m) {
    m.def("row_roundtrip", &carma::tests::test_row_roundtrip, "Test row_roundtrip");
}

void bind_test_col_roundtrip(py::module& m) {
    m.def("col_roundtrip", &carma::tests::test_col_roundtrip, "Test col_roundtrip");
}

void bind_test_cube_roundtrip(py::module& m) {
    m.def("cube_roundtrip", &carma::tests::test_cube_roundtrip, "Test cube_roundtrip");
}
