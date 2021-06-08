#include "ols.h"

#include <carma>
#include <armadillo>
#include <pybind11/pybind11.h>
#include <pybind11/numpy.h>
#include <pybind11/pytypes.h>

py::tuple ols(arma::colvec& y, arma::mat& X) {
    int n = X.n_rows, k = X.n_cols;

    arma::colvec coeffs = arma::solve(X, y);
    arma::colvec resid = y - X * coeffs;

    double sig2 = arma::as_scalar(arma::trans(resid) * resid / (n-k));
    arma::colvec std_errs = arma::sqrt(sig2 * arma::diagvec( arma::inv(arma::trans(X)*X)) );

    return py::make_tuple(
        carma::col_to_arr(coeffs),
        carma::col_to_arr(std_errs)
    );
}

void bind_ols(py::module &m) {
    m.def(
        "ols",
        &ols,
        R"pbdoc(
            Example function performing OLS.

            Parameters
            ----------
            arr : np.array
                input array

            Returns
            -------
            coeffs: np.ndarray
                coefficients
            std_err : np.ndarray
                standard error on the coefficients
        )pbdoc",
        py::arg("y"),
        py::arg("x")
    );
}
