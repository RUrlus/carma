#include "mre.h"

namespace py = pybind11;

PYBIND11_MODULE(pymod, m) {
  m.def("test_runtime_error", []() {
    throw std::runtime_error("some error");
  });
};
