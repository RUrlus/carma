#include "manual_conversion.h"

py::array_t<double> manual_example(py::array_t<double> & arr) {
    // convert to armadillo matrix without copying.
    arma::Mat<double> mat = carma::arr_to_mat<double>(arr);

    // normally you do something useful here ...
    arma::Mat<double> result = arma::Mat<double>(arr.shape(0), arr.shape(1), arma::fill::randu);

    // convert to Numpy array and return
    return carma::mat_to_arr(result);
}


void update_example(py::array_t<double> & arr) {
    // convert to armadillo matrix without copying.
    arma::Mat<double> mat = carma::arr_to_mat<double>(arr);

    // normally you do something useful here with mat ...
    mat += arma::Mat<double>(arr.shape(0), arr.shape(1), arma::fill::randu);

    // update Numpy array buffer
    carma::update_array(mat, arr);
}

void bind_manual_example(py::module &m) {
    m.def(
        "manual_example",
        &manual_example,
        R"pbdoc(
            Example function for manual conversion.

            Parameters
            ----------
            mat : np.array
                input array
        )pbdoc",
        py::arg("arr")
    );
}

void bind_update_example(py::module &m) {
    m.def(
        "update_example",
        &update_example,
        R"pbdoc(
            Example function for update conversion.

            Parameters
            ----------
            mat : np.array
                input array
        )pbdoc",
        py::arg("arr")
    );
}
