Examples
########

On a high level CARMA provides two ways to work with Numpy arrays and Armadillo,
see the :doc:`Function specifications <carma>` section for details about the available functions and the examples directory for runnable examples.

Manual conversion
-----------------

The easiest way to use CARMA is manual conversion, it gives you the most control over when to copy or not.
You pass a Numpy array as an input and/or as the return type and call the respective conversion function.

.. warning:: Carma will avoid copying by default so make sure not to return the memory of the input array without copying. If you don't copy out, the memory is aliased by both the input and output arrays which will cause a segfault.

Borrow
******

.. code-block:: c++

    #include <carma>
    #include <armadillo>
    #include <pybind11/pybind11.h>
    #include <pybind11/numpy.h>
    
    py::array_t<double> manual_example(py::array_t<double> & arr) {
        // convert to armadillo matrix without copying.
        // Note the size of the matrix cannot be changed when borrowing
        arma::Mat<double> mat = carma::arr_to_mat<double>(arr);
    
        // normally you do something useful here ...
        arma::Mat<double> result = arma::Mat<double>(arr.shape(0), arr.shape(1), arma::fill::randu);
    
        // convert to Numpy array and copy out
        return carma::mat_to_arr(result, true);
    }

Transfer ownership
******************

If you want to transfer ownership to the C++ side you can use:

.. code-block:: c++

    #include <carma>
    #include <armadillo>
    #include <pybind11/pybind11.h>
    #include <pybind11/numpy.h>
    
    py::array_t<double> steal_array(py::array_t<double> & arr) {
        // convert to armadillo matrix
        arma::Mat<double> mat = carma::arr_to_mat<double>(std::move(arr));
        // armadillo now owns and manages the lifetime of the memory
        // if you want to return you don't need to copy out
        return mat_to_arr(mat);
    }

    py::array_t<double> copy_array(py::array_t<double> & arr) {
        // convert to armadillo matrix
        arma::Mat<double> mat = carma::arr_to_mat<double>(arr, true);
        // armadillo now owns and manages the lifetime of the memory
        // if you want to return you don't need to copy out
        return mat_to_arr(mat);
    }

    py::array_t<double> copy_const_array(const py::array_t<double> & arr) {
        // convert to armadillo matrix
        arma::Mat<double> mat = carma::arr_to_mat<double>(arr);
        // armadillo now owns and manages the lifetime of the memory
        // if you want to return you don't need to copy out
        return mat_to_arr(mat);
    }

Automatic conversion
--------------------

For automatic conversion you specify the desired Armadillo type for either or both the return type and the function parameter.
When calling the function from Python, Pybind11 will call CARMA's type caster when a Numpy array is passed or returned, see :ref:`Return policies` for details.

.. warning:: Make sure to include `carma` in every compilation unit that makes use of the type caster, not including it results in undefined behaviour.

.. code-block:: c++

    #include <carma>
    #include <armadillo>
    #include <pybind11/pybind11.h>
    #include <pybind11/numpy.h>
    
    arma::Mat<double> automatic_example(arma::Mat<double> & mat) {
        // normally you do something useful here with mat ...
        arma::Mat<double> rand = arma::Mat<double>(mat.n_rows, mat.n_cols, arma::fill::randu);
    
        arma::Mat<double> result = mat + rand;
        // type caster will take care of casting `result` to a Numpy array.
        return result;
    }

.. warning::
    
    The automatic conversion will **not** copy the Numpy array's memory when converting to Armadillo objects.
    When converting back to Numpy arrays the memory will **not** be copied out
    by default. You shoud specify ``return_value_policy::copy`` if you want to
    return the input array.

ArrayStore
----------

There are use-cases where you would want to keep the data in C++ and only return when requested.
For example, you write an Ordinary Least Squares (OLS) class and you want to store the residuals, covariance matrix, ... in C++ for when additional tests need to be run on the values without converting back and forth.

ArrayStore is a convenience class that provides conversion methods back and forth.
It is intended to be used as an attribute as below:

.. warning::
    
    The ArrayStore owns the data, the returned numpy arrays are views that
    are tied to the lifetime of ArrayStore.

.. code-block:: c++

    #include <armadillo>
    #include <carma>
    #include <pybind11/pybind11.h>
    #include <pybind11/numpy.h>
    
    class ExampleClass {
        private:
            carma::ArrayStore<arma::Mat<double>> _x;
            carma::ArrayStore<arma::Mat<double>> _y;
    
        public:
            ExampleClass(py::array_t<double> & x, py::array_t<double> & y) :
            // copy the array and store it as an Armadillo matrix
            _x{carma::ArrayStore<arma::Mat<double>>(x, true)},
            // steal the array and store it as an Armadillo matrix
            _y{carma::ArrayStore<arma::Mat<double>>(y, false)},
    
            py::array_t<double> member_func() {
                // normallly you would something useful here
                _x.mat += _y.mat;
                // return mutable view off arma matrix
                return _x.get_view(true);
            }
    };

    void bind_exampleclass(py::module &m) {
        py::class_<ExampleClass>(m, "ExampleClass")
            .def(py::init<py::array_t<double> &, py::array_t<double> &>(), R"pbdoc(
                Initialise ExampleClass.
    
                Parameters
                ----------
                arr1: np.ndarray
                    array to be stored in armadillo matrix
                arr2: np.ndarray
                    array to be stored in armadillo matrix
            )pbdoc")
            .def("member_func", &ExampleClass::member_func, R"pbdoc(
                Compute ....
            )pbdoc");
    }



Ordinary Least Squares
----------------------

Combining the above approaches to compute the Ordinary Least Squares:

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

Which can be called using:

.. code-block:: c++

    y = np.linspace(1, 100, num=100) + np.random.normal(0, 0.5, 100)
    X = np.hstack((np.ones(100)[:, None], np.arange(1, 101)[:, None]))
    coeff, std_err = carma.ols(X, y)

The `repository <https://github.com/RUrlus/carma/tree/stable/examples>`_ contains tests, examples and CMake build instructions that can be used as an reference.
