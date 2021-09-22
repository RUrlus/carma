#include "mre.h"

namespace py = pybind11;

class MatrixHolder {
 public:
  explicit MatrixHolder(size_t d) {
    A = arma::Mat<double>(d, d, arma::fill::eye);
    std::cerr << "filled arma matrix\n";
  }
  arma::Mat<double> A;
};

PYBIND11_MODULE(pymod, m) {
  py::class_<MatrixHolder>(m, "MH").def(py::init<size_t>())
      .def_readwrite("A", &MatrixHolder::A);
};
