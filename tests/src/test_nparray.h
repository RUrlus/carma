#include <pybind11/numpy.h>
#include <pybind11/pybind11.h>
namespace py = pybind11;

#include <carma/nparray.h>

namespace carma { namespace tests {

    long test_flat_reference_long(py::array_t<long> & arr, size_t index);
    double test_flat_reference_double(py::array_t<double> & arr, size_t index);
    long test_mutable_flat_reference_long(
        py::array_t<long> & arr, size_t index, long value
    );
    double test_mutable_flat_reference_double(
        py::array_t<double> & arr, size_t index, double value
    );


    long test_flat_reference_long(py::array_t<long> & arr, size_t index);
    double test_flat_reference_double(py::array_t<double> & arr, size_t index);
    long test_mutable_flat_reference_long(
        py::array_t<long> & arr, size_t index, long value
    );
    double test_mutable_flat_reference_double(
        py::array_t<double> & arr, size_t index, double value
    );

} /* tests */ } /* carma */

void bind_test_is_f_contiguous(py::module &m);
void bind_test_is_c_contiguous(py::module &m);
void bind_test_is_writable(py::module &m);
void bind_test_is_owndata(py::module &m);
void bind_test_is_aligned(py::module &m);
void bind_test_flat_reference(py::module &m);
void bind_test_mutable_flat_reference(py::module &m);
