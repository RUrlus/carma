.. role:: cmake(code)
   :language: cmake

Introduction
############

CARMA provides fast bidirectional conversions between Numpy_ arrays and Armadillo_ matrices, vectors and cubes, much like RcppArmadillo_ does for R and Armadillo.

The library extends the impressive Pybind11_ library with support for Armadillo.
For details on Pybind11 and Armadillo refer to their respective documentation `[1] <https://pybind11.readthedocs.io/en/stable/intro.html>`_, `[2] <http://arma.sourceforge.net/docs.html>`_.

Installation
++++++++++++
CARMA is a header only library that relies on two other header only libraries, Armadillo and Pybind11.

CARMA can be integrated in a CMake build using :cmake:`ADD_SUBDIRECTORY(<path_to_carma>)` which provides an interface library target ``carma`` that has been linked with Python, Numpy, Pybind11 and Armadillo.

To link with CARMA:

.. code-block:: cmake

    ADD_SUBDIRECTORY(extern/carma)
    TARGET_LINK_LIBRARIES(<your_target> PRIVATE carma)

CARMA and Armadillo can then be included using:

.. code-block:: c++

    #include <carma>
    #include <armadillo>

CARMA provides a number of configurations that can be set in the ``carma_config.cmake`` file at the root of the directory or passed to CMake, see :ref:`Configuration` and :ref:`Build configuration` sections for details.

Requirements
++++++++++++

CARMA v0.5 requires a compiler with support for C++14 and supports:

- Python 3.6 -- 3.9
- Numpy >= 1.14
- Pybind11 v2.6.0 -- v2.6.2
- Armadillo >= 10.5.2

CARMA makes use of Armadillo's ``ARMA_ALIEN_MEM_ALLOC`` and ``ARMA_ALIEN_MEM_FREE`` functionality introduced in version 10.5.2 to use Numpy's (de)allocator.

OLS example
+++++++++++

A brief example on how conversion can be achieved. We convert the Numpy arrays X and y in using automatic casting (type-caster) and using manual conversion on the way out.

.. code-block:: c++

    #include <carma>
    #include <armadillo>
    #include <pybind11/pybind11.h>
    #include <pybind11/numpy.h>
    #include <pybind11/pytypes.h>
    
    py::tuple ols(arma::mat& X, arma::colvec& y) {
        // We borrow the data underlying the numpy arrays
        int n = X.n_rows, k = X.n_cols;
    
        arma::colvec coeffs = arma::solve(X, y);
        arma::colvec resid = y - X * coeffs;
    
        double sig2 = arma::as_scalar(arma::trans(resid) * resid / (n-k));
        arma::colvec std_errs = arma::sqrt(sig2 * arma::diagvec( arma::inv(arma::trans(X)*X)) );
    
        // We take ownership of the memory from the armadillo objects and
        // return to python as a tuple containing two Numpy arrays.
        return py::make_tuple(
            carma::col_to_arr(coeffs),
            carma::col_to_arr(std_errs)
        );
    }

    // adapted from https://gallery.rcpp.org/articles/fast-linear-model-with-armadillo/

Considerations
++++++++++++++

In order to achieve fast conversions the default behaviour is avoid copying both from and to Numpy whenever possible and reasonable.
This allows very low overhead conversions but it impacts memory safety and requires user vigilance.

If you intend to return the memory of the input array back as another array, you must make sure to either copy or steal the memory on the conversion in or copy the memory out.
If you don't the memory will be aliased by the two Numpy arrays and bad things will happen.

A second consideration is memory layout. Armadillo is optimised for column-major (Fortran order) memory whereas Numpy defaults to row-major (C order).
The default behaviour is to automatically convert, read copy, C-order arrays to F-order arrays upon conversion to Armadillo. Users should note that the library will not convert back to C-order when returning.

For details see the :doc:`Memory Management <memory_management>` section.

About
#####

This project was created by Ralph Urlus. Significant improvements to the project have been contributed by `Pascal H. <https://github.com/hpwxf>`_

License
+++++++

`carma` is provided under a Apache 2.0 license that can be found in the LICENSE file. By using, distributing, or contributing to this project, you agree to the terms and conditions of this license.

.. _numpy: https://numpy.org
.. _rcpparmadillo: https://github.com/RcppCore/RcppArmadillo
.. _pybind11: https://pybind11.readthedocs.io/en/stable/intro.html
.. _armadillo: http://arma.sourceforge.net/docs.html
