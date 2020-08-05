#include <pybind11/numpy.h>
#include <pybind11/pybind11.h>
#include <armadillo>

#include <carma/carma.h>

namespace py = pybind11;

namespace carma {
namespace tests {
// type_caster_in
double test_tc_in_mat(const arma::Mat<double>& mat);
double test_tc_in_row(const arma::Row<double>& mat);
double test_tc_in_col(const arma::Col<double>& mat);
double test_tc_in_cube(const arma::Cube<double>& mat);

// type_caster_out
arma::Mat<double> test_tc_out_mat(const py::array_t<double>& arr);
arma::Mat<double> test_tc_out_mat_rvalue(const py::array_t<double>& arr);
arma::Row<double> test_tc_out_row(const py::array_t<double>& arr);
arma::Row<double> test_tc_out_row_rvalue(const py::array_t<double>& arr);
arma::Col<double> test_tc_out_col(const py::array_t<double>& arr);
arma::Col<double> test_tc_out_col_rvalue(const py::array_t<double>& arr);
arma::Cube<double> test_tc_out_cube(const py::array_t<double>& arr);
arma::Cube<double> test_tc_out_cube_rvalue(const py::array_t<double>& arr);
}  // namespace tests
}  // namespace carma

// type_caster_in
void bind_test_tc_in_mat(py::module& m);
void bind_test_tc_in_row(py::module& m);
void bind_test_tc_in_col(py::module& m);
void bind_test_tc_in_cube(py::module& m);
void bind_test_tc_out_mat(py::module& m);
void bind_test_tc_out_row(py::module& m);

// type_caster_out
void bind_test_tc_out_mat(py::module& m);
void bind_test_tc_out_mat_rvalue(py::module& m);
void bind_test_tc_out_row(py::module& m);
void bind_test_tc_out_row_rvalue(py::module& m);
void bind_test_tc_out_col(py::module& m);
void bind_test_tc_out_col_rvalue(py::module& m);
void bind_test_tc_out_cube(py::module& m);
void bind_test_tc_out_cube_rvalue(py::module& m);
