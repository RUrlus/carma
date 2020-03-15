#include <pybind11/pybind11.h>
// include numpy header for usage of array_t
#include <pybind11/numpy.h>

namespace py = pybind11;

#include "manual_conversion.h"
#include "automatic_conversion.h"

PYBIND11_MODULE(carma_examples, m) {
    bind_manual_example(m);
    bind_update_example(m);
    bind_automatic_example(m);
}
