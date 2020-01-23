#include "test_nparray.h"

namespace carma { namespace tests {

    long test_flat_reference_long(py::array_t<long> & arr, size_t index) {
        flat_reference<long> m_ptr = flat_reference<long>(arr);
        return m_ptr[index];
    }

    double test_flat_reference_double(py::array_t<double> & arr, size_t index) {
        flat_reference<double> m_ptr = flat_reference<double>(arr);
        return m_ptr[index];
    }

    long test_mutable_flat_reference_long(
        py::array_t<long> & arr, size_t index, long value
    ) {
        mutable_flat_reference<long> m_ptr = mutable_flat_reference<long>(arr);
        m_ptr[index] = value;
        return m_ptr[index];
    }

    double test_mutable_flat_reference_double(
        py::array_t<double> & arr, size_t index, double value
    ) {
        mutable_flat_reference<double> m_ptr = mutable_flat_reference<double>(arr);
        m_ptr[index] = value;
        return m_ptr[index];
    }

} /* tests */
} /* carma */

void bind_test_is_f_contiguous(py::module &m) {
    m.def(
        "is_f_contiguous",
        [](py::array_t<double> & arr) {return carma::is_f_contiguous(arr);},
        "Test is F contiguous"
    );
}

void bind_test_is_c_contiguous(py::module &m) {
    m.def(
        "is_c_contiguous",
        [](py::array_t<double> & arr) {return carma::is_c_contiguous(arr);},
        "Test is C contiguous"
    );
}

void bind_test_is_writable(py::module &m) {
    m.def(
        "is_writable",
        [](py::array_t<double> & arr) {return carma::is_writable(arr);},
        "Test is writable"
    );
}

void bind_test_is_owndata(py::module &m) {
    m.def(
        "is_owndata",
        [](py::array_t<double> & arr) {return carma::is_owndata(arr);},
        "Test is owndata"
    );
}

void bind_test_is_aligned(py::module &m) {
    m.def(
        "is_aligned",
        [](py::array_t<double> & arr) {return carma::is_aligned(arr);},
        "Test is aligned"
    );
}

void bind_test_flat_reference(py::module &m) {
    m.def(
        "flat_reference",
        &carma::tests::test_flat_reference_double,
        "Test flat_reference "
    );
    m.def(
        "flat_reference",
        &carma::tests::test_flat_reference_long,
        "Test flat_reference "
    );
}

void bind_test_mutable_flat_reference(py::module &m) {
    m.def(
        "mutable_flat_reference",
        &carma::tests::test_mutable_flat_reference_double,
        "Test mutable_flat_reference "
    );
    m.def(
        "mutable_flat_reference",
        &carma::tests::test_mutable_flat_reference_long,
        "Test mutable_flat_reference "
    );
}
