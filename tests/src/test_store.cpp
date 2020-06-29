#include <limits>
#include "test_store.h"

namespace carma { namespace tests {

double test_ArrayStore_get_mat() {
    arma::Mat<double> mat_in = arma::ones(100, 1);
    ArrayStore<double> store = ArrayStore<double>(mat_in);
    arma::Mat<double> mat_out = store.mat;
    return std::fabs(arma::accu(mat_out) - arma::accu(mat_in));
} /* test_ArrayStore_get_mat */

double test_ArrayStore_get_mat_rvalue() {
    arma::Mat<double> mat_in = arma::ones(100, 1);
    arma::Mat<double> ref_mat = arma::mat(mat_in.memptr(), 100, 1);
    ArrayStore<double> store = (
        ArrayStore<double>(std::forward<arma::mat>(mat_in))
    );
    arma::Mat<double> mat_out = store.mat;
    return std::fabs(arma::accu(mat_out) - arma::accu(ref_mat));
} /* test_ArrayStore_get_mat */

py::array_t<double> test_ArrayStore_get_view(bool writeable) {
    arma::Mat<double> mat_in = arma::ones(100, 1);
    ArrayStore<double> store = ArrayStore<double>(mat_in);
    return store.get_view(writeable);
} /* test_ArrayStore_get_mat_const */

} /* tests */ } /* carma */

void bind_test_ArrayStore_get_mat(py::module &m) {
    m.def(
        "test_ArrayStore_get_mat",
        &carma::tests::test_ArrayStore_get_mat,
        "Test ArrayStore"
    );
}

void bind_test_ArrayStore_get_mat_rvalue(py::module &m) {
    m.def(
        "test_ArrayStore_get_mat_rvalue",
        &carma::tests::test_ArrayStore_get_mat_rvalue,
        "Test ArrayStore"
    );
}

void bind_test_ArrayStore_get_view(py::module &m) {
    m.def(
        "test_ArrayStore_get_view",
        &carma::tests::test_ArrayStore_get_view,
        "Test ArrayStore"
    );
}
