#ifndef TESTS_SRC_TEST_NPARRAY_H_
#define TESTS_SRC_TEST_NPARRAY_H_
#include <pybind11/numpy.h>
#include <pybind11/pybind11.h>
#include <carma>
#include <stdexcept>

namespace py = pybind11;

void bind_test_is_f_contiguous(py::module& m);
void bind_test_is_c_contiguous(py::module& m);
void bind_test_is_writeable(py::module& m);
void bind_test_is_owndata(py::module& m);
void bind_test_is_aligned(py::module& m);
void bind_test_set_not_owndata(py::module& m);
void bind_test_set_not_writeable(py::module& m);
void bind_test_is_well_behaved(py::module& m);
void bind_test_exception_flow(py::module&m);
void bind_test_conversion_error(py::module&m);

#endif  //  TESTS_SRC_TEST_NPARRAY_H_
