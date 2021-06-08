#include <carma>
#include <armadillo>
#include <pybind11/pybind11.h>
#include <pybind11/numpy.h>
#include <pybind11/pytypes.h>

py::tuple ols(arma::colvec& y, arma::mat& X);
void bind_ols(py::module &m);
