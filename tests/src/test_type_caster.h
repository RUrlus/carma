#include <pybind11/numpy.h>
#include <pybind11/pybind11.h>
#include <armadillo>
namespace py = pybind11;

#include <carma/carma.h>

namespace carma {
namespace tests {
// type_caster_in
double test_tc_in_mat(arma::Mat<double>& mat);
double test_tc_in_row(arma::Row<double>& mat);
double test_tc_in_col(arma::Col<double>& mat);
double test_tc_in_cube(arma::Cube<double>& mat);
double test_tc_in_fixed_vec3(arma::vec3& mat);
double test_tc_in_fixed_vec4(arma::vec4& mat);
double test_tc_in_fixed_mat33(arma::mat33& mat);
double test_tc_in_fixed_rowvec3(arma::rowvec3& mat);

// type_caster_out
arma::Mat<double> test_tc_out_mat(py::array_t<double>& arr);
arma::Mat<double> test_tc_out_mat_rvalue(py::array_t<double>& arr);
arma::Row<double> test_tc_out_row(py::array_t<double>& arr);
arma::Row<double> test_tc_out_row_rvalue(py::array_t<double>& arr);
arma::Col<double> test_tc_out_col(py::array_t<double>& arr);
arma::Col<double> test_tc_out_col_rvalue(py::array_t<double>& arr);
arma::Cube<double> test_tc_out_cube(py::array_t<double>& arr);
arma::Cube<double> test_tc_out_cube_rvalue(py::array_t<double>& arr);
}  // namespace tests
}  // namespace carma

// type_caster_in
void bind_test_tc_in_mat(py::module& m);
void bind_test_tc_in_row(py::module& m);
void bind_test_tc_in_col(py::module& m);
void bind_test_tc_in_cube(py::module& m);
void bind_test_tc_in_fixed_vec3(py::module& m);
void bind_test_tc_in_fixed_vec4(py::module& m);
void bind_test_tc_in_fixed_mat33(py::module& m);
void bind_test_tc_in_fixed_rowvec3(py::module& m);
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
