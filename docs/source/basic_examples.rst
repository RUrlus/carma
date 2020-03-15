Examples
########

On a high level `carma` provides three ways to work Numpy arrays in Armadillo:
See the :doc:`Function specifications <carma>` section for details about the available functions and the examples directory for runnable examples.

Manual conversion
+++++++++++++++++

The easiest way to use `carma` is manual conversion, it gives you the most control over when to copy or not.
You pass a Numpy array as an argument and/or as the return type and call the respective conversion function.

.. warning:: Carma will avoid copying by default so make sure not to return the memory of the input array without copying or use `update_array`.

.. code-block:: c++

    #include <armadillo>
    #include <carma/carma.h>
    #include <pybind11/pybind11.h>
    #include <pybind11/numpy.h>
    
    py::array_t<double> manual_example(py::array_t<double> & arr) {
        // convert to armadillo matrix without copying.
        arma::Mat<double> mat = carma::arr_to_mat<double>(arr);
    
        // normally you do something useful here ...
        arma::Mat<double> result = arma::Mat<double>(arr.shape(0), arr.shape(1), arma::fill::randu);
    
        // convert to Numpy array and return
        return carma::mat_to_arr(result);
    }

Update array
++++++++++++

.. code-block:: c++

    #include <armadillo>
    #include <carma/carma.h>
    #include <pybind11/pybind11.h>
    #include <pybind11/numpy.h>
    
    void update_example(py::array_t<double> & arr) {
        // convert to armadillo matrix without copying.
        arma::Mat<double> mat = carma::arr_to_mat<double>(arr);
    
        // normally you do something useful here with mat ...
        mat += arma::Mat<double>(arr.shape(0), arr.shape(1), arma::fill::randu);
    
        // update Numpy array buffer
        carma::update_array(mat, arr);
    }

Automatic conversion
++++++++++++++++++++

For automatic conversion you specify the desired Armadillo type for either or both the return type and the function parameter.
When calling the function from Python, Pybind11 will call `carma`'s type caster when a Numpy array is passed or returned.

.. warning:: Make sure to include `carma` in every compilation unit that makes use of the type caster, not including it results in undefined behaviour.

.. code-block:: c++

    #include <armadillo>
    #include <carma/carma.h>
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
    When converting back to Numpy arrays the memory will **not** be copied when converting back from matrices but **will be** copied from a vector or cube.
    See :doc:`Memory Management <memory_management>` for details.
