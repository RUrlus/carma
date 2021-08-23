#include <carma>
#include <armadillo>
#include <pybind11/pybind11.h>
#include <pybind11/numpy.h>

namespace py = pybind11;

void bind_test_is_f_contiguous(py::module& m) {
    m.def(
        "is_f_contiguous",
        [](const py::array_t<double>& arr) { return carma::is_f_contiguous(arr); },
        "Test is F contiguous");
}

py::array_t<double> test_mat_roundtrip(py::array_t<double>& arr) {
    // call function to be tested
    arma::Mat<double> M = carma::arr_to_mat<double>(std::move(arr));
    return carma::mat_to_arr(M);
}

void bind_test_mat_roundtrip(py::module& m) {
    m.def("mat_roundtrip", &test_mat_roundtrip, "Test mat_roundtrip");
}

PYBIND11_MODULE(integration_test_carma, m) {
    bind_test_is_f_contiguous(m);
    bind_test_mat_roundtrip(m);
}
