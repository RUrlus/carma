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

PYBIND11_MODULE(integration_test_carma, m) {
    bind_test_is_f_contiguous(m);
}
