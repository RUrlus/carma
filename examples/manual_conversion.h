#include <pybind11/pybind11.h>
#include <pybind11/numpy.h>

#include <armadillo>
#include <carma>

void update_example(py::array_t<double> & arr);
py::array_t<double> manual_example(py::array_t<double> & arr);

void bind_manual_example(py::module &m);
