#include <pybind11/numpy.h>
#include <pybind11/pybind11.h>
namespace py = pybind11;

#include <carma/nparray.h>

void bind_test_is_f_contiguous(py::module &m);
void bind_test_is_c_contiguous(py::module &m);
void bind_test_is_writable(py::module &m);
void bind_test_is_owndata(py::module &m);
void bind_test_is_aligned(py::module &m);
