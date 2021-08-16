#################
Manual conversion
#################

CARMA provides a set of functions for manual conversion of Numpy arrays and Armadillo matrices.
Manual conversion should be used when fine grained control of memory is required.

Numpy to Armadillo
******************

Functions to convert Numpy arrays to Armadillo matrices, vectors or cubes

Matrix
------

.. function:: arma::Mat<T> arr_to_mat(py::array<T>& arr, bool copy=false)

    Convert Numpy array to Armadillo matrix.
    When borrowing the the array, copy=false, if it is not well-behaved we perform a in-place swap to a well behaved array
    
    If the array is 1D we create a column oriented matrix (N, 1)
    
    :param arr: numpy array to be converted
    :param copy: copy the memory of the array, default is false
    :exception `runtime_error`: if `n_dims` > 2 or memory is not initialised (nullptr)
    :exception `runtime_error`: if copy is false and the array is not well-behaved and not writable as this prevents an inplace-swap.

.. function:: arma::Mat<T> arr_to_mat(const py::array<T>& arr)

    Copy the memory of the Numpy array in the conversion to Armadillo matrix.
    
    If the array is 1D we create a column oriented matrix (N, 1)
    
    :param arr: numpy array to be converted
    :exception `runtime_error`: if `n_dims` > 2 or memory is not initialised (nullptr)

.. function:: arma::Mat<T> arr_to_mat(py::array<T>&& arr)

    Steal the memory of the Numpy array in the conversion to Armadillo matrix.
    If the memory is not well-behaved we steal a copy of the array

    If the array is 1D we create a column oriented matrix (N, 1)
    
    :param arr: numpy array to be converted
    :exception `runtime_error`: if `n_dims` > 2 or memory is not initialised (nullptr)

.. function:: const arma::Mat<T> arr_to_mat_view(const py::array<T>& arr)

    Copy the memory of the Numpy array in the conversion to Armadillo matrix if not well behaved otherwise borrow.
    
    If the array is 1D we create a column oriented matrix (N, 1)
    
    :param arr: numpy array to be converted
    :exception `runtime_error`: if `n_dims` > 2 or memory is not initialised (nullptr)

Vector
------

.. function:: arma::Col<T> arr_to_col(py::array<T>& arr, bool copy=false)

    Convert Numpy array to Armadillo column.
    When borrowing the the array, copy=false, if it is not well-behaved we perform a in-place swap to a well behaved array

    :param arr: numpy array to be converted
    :param copy: copy the memory of the array, default is false
    :exception `runtime_error`: if `n_cols` > 1 or memory is not initialised (nullptr)
    :exception `runtime_error`: if copy is false and the array is not well-behaved and not writable as this prevents an inplace-swap.

.. function:: arma::Col<T> arr_to_col(const py::array<T>& arr)

    Copy the memory of the Numpy array in the conversion to Armadillo column.
    
    :param arr: numpy array to be converted
    :exception `runtime_error`: if `n_cols` > 1 or memory is not initialised (nullptr)

.. function:: arma::Col<T> arr_to_col(py::array<T>&& arr)

    Steal the memory of the Numpy array in the conversion to Armadillo column.
    If the memory is not well-behaved we steal a copy of the array
    
    :param arr: numpy array to be converted
    :exception `runtime_error`: if `n_cols` > 1 or memory is not initialised (nullptr)

.. function:: const arma::Col<T> arr_to_col_view(const py::array<T>& arr)

    Create a read-only view on the array as a Armadillo Col.
    Copy the memory of the Numpy array in the conversion to Armadillo column if not well_behaved otherwise borrow the array.
    
    :param arr: numpy array to be converted
    :exception `runtime_error`: if `n_cols` > 1 or memory is not initialised (nullptr)

.. function:: arma::Row<T> arr_to_row(py::array<T>& arr, bool copy=false)

    Convert Numpy array to Armadillo row.
    When borrowing the the array, copy=false, if it is not well-behaved we perform a in-place swap to a well behaved array
    
    :param arr: numpy array to be converted
    :param copy: copy the memory of the array, default is false
    :exception `runtime_error`: if `n_rows` > 1 or memory is not initialised (nullptr)
    :exception `runtime_error`: if copy is false and the array is not well-behaved and not writable as this prevents an inplace-swap.

.. function:: arma::Row<T> arr_to_col(const py::array<T>& arr)

    Copy the memory of the Numpy array in the conversion to Armadillo row.
    
    :param arr: numpy array to be converted
    :exception `runtime_error`: if `n_rows` > 1 or memory is not initialised (nullptr)

.. function:: arma::Row<T> arr_to_col(py::array<T>&& arr)

    Steal the memory of the Numpy array in the conversion to Armadillo row.
    If the memory is not well-behaved we steal a copy of the array
    
    :param arr: numpy array to be converted
    :exception `runtime_error`: if `n_cols` > 1 or memory is not initialised (nullptr)

.. function:: const arma::Row<T> arr_to_col_view(const py::array<T>& arr)

    Create a read-only view on the array as a Armadillo Col.
    Copy the memory of the Numpy array if not well_behaved otherwise borrow the array.
    
    :param arr: numpy array to be converted
    :exception `runtime_error`: if `n_rows` > 1 or memory is not initialised (nullptr)

Cube
----

.. function:: arma::Cube<T> arr_to_cube(py::array<T>& arr, bool copy=false)

    Convert Numpy array to Armadillo Cube.
    When borrowing the the array, copy=false, if it is not well-behaved we perform a in-place swap to a well behaved array
    
    :param arr: numpy array to be converted
    :param copy: copy the memory of the array, default is false
    :exception `runtime_error`: if `n_dims` < 3 or memory is not initialised (nullptr)
    :exception `runtime_error`: if copy is false and the array is not well-behaved and not writable as this prevents an inplace-swap.

.. function:: arma::Cube<T> arr_to_cube(const py::array<T>& arr)

    Copy the memory of the Numpy array in the conversion to Armadillo Cube.
    
    :param arr: numpy array to be converted
    :exception `runtime_error`: if `n_dims` < 3 or memory is not initialised (nullptr)

.. function:: arma::Cube<T> arr_to_cube(py::array<T>&& arr)

    Steal the memory of the Numpy array in the conversion to Armadillo Cube.
    If the memory is not well-behaved we steal a copy of the array
    
    :param arr: numpy array to be converted
    :exception `runtime_error`: if `n_dims` < 3 or memory is not initialised (nullptr)

.. function:: const arma::Cube<T> arr_to_cube_view(const py::array<T>& arr)

    Create a read-only view on the array as a Armadillo Cube.
    Copy the memory of the Numpy array if not well_behaved otherwise borrow the array.
    
    :param arr: numpy array to be converted
    :exception `runtime_error`: if `n_dims` < 3 or memory is not initialised (nullptr)

to_arma
-------

``to_arma`` is a convenience wrapper around the ``arr_to_*`` functions and has the same behaviour and rules. For example,

.. code-block:: c++

    arma::Mat<double> mat = to_arma::from<arma::Mat<double>>(arr, copy=false);

.. function:: template <typename armaT> armaT to_arma::from(const py::array_t<eT>& arr)

.. function:: template <typename armaT> armaT to_arma::from(py::array_t<eT>& arr, bool copy)

.. function:: template <typename armaT> armaT to_arma::from( py::array_t<eT>&& arr)

Armadillo to Numpy
******************

This section documents the functions to convert Armadillo matrices, vectors or cubes to Numpy arrays. 

Matrix
------

.. function:: py::array_t<T> mat_to_arr(arma::Mat<T>& src, bool copy=false)

    Convert Armadillo matrix to Numpy array, note the returned array will have column contiguous memory (F-order)
    
    :note: the returned array will have F order memory.
    :param src: armadillo object to be converted.
    :param copy: copy the memory of the array, default is false which steals the memory

.. function:: py::array_t<T> mat_to_arr(const arma::Mat<T>& src)

    Copy the memory of the Armadillo matrix in the conversion to Numpy array.
    
    :note: the returned array will have F order memory.
    :param src: armadillo object to be converted.

.. function:: py::array_t<T> mat_to_arr(arma::Mat<T>&& src)

    Steal the memory of the Armadillo matrix in the conversion to Numpy array.
    
    :note: the returned array will have F order memory.
    :param src: armadillo object to be converted.

.. function:: py::array_t<T> mat_to_arr(arma::Mat<T>* src, bool copy=false)

    Convert Armadillo matrix to Numpy array.
    
    :note: the returned array will have F order memory.
    :param src: armadillo object to be converted.
    :param copy: copy the memory of the array, default is false

Vector
------

.. function:: py::array_t<T> col_to_arr(arma::Col<T>& src, bool copy=false)

    Convert Armadillo col to Numpy array, note the returned array will have column contiguous memory (F-order)
    
    :note: the returned array will have F order memory.
    :param src: armadillo object to be converted.
    :param copy: copy the memory of the array, default is false which steals the memory

.. function:: py::array_t<T> col_to_arr(const arma::Col<T>& src)

    Copy the memory of the Armadillo Col in the conversion to Numpy array.
    
    :note: the returned array will have F order memory.
    :param src: armadillo object to be converted.

.. function:: py::array_t<T> col_to_arr(arma::Col<T>&& src)

    Steal the memory of the Armadillo Col in the conversion to Numpy array.
    
    :note: the returned array will have F order memory.
    :param src: armadillo object to be converted.

.. function:: py::array_t<T> col_to_arr(arma::Col<T>* src, bool copy=false)

    Convert Armadillo Col to Numpy array.
    
    :note: the returned array will have F order memory.
    :param src: armadillo object to be converted.
    :param copy: copy the memory of the array, default is false

.. function:: py::array_t<T> row_to_arr(arma::Row<T>& src, bool copy=false)

    Convert Armadillo Row to Numpy array, note the returned array will have column contiguous memory (F-order)
    
    :note: the returned array will have F order memory.
    :param src: armadillo object to be converted.
    :param copy: copy the memory of the array, default is false which steals the memory

.. function:: py::array_t<T> row_to_arr(const arma::Row<T>& src)

    Copy the memory of the Armadillo Row in the conversion to Numpy array.
    
    :note: the returned array will have F order memory.
    :param src: armadillo object to be converted.

.. function:: py::array_t<T> row_to_arr(arma::Row<T>&& src)

    Steal the memory of the Armadillo Row in the conversion to Numpy array.
    
    :note: the returned array will have F order memory.
    :param src: armadillo object to be converted.

.. function:: py::array_t<T> row_to_arr(arma::Row<T>* src, bool copy=false)

    Convert Armadillo Row to Numpy array.
    
    :note: the returned array will have F order memory.
    :param src: armadillo object to be converted.
    :param copy: copy the memory of the array, default is false

Cube
----

.. function:: py::array_t<T> cube_to_arr(arma::Cube<T>& src, bool copy=false)

    Convert Armadillo Cube to Numpy array, note the returned array will have column contiguous memory (F-order)
    
    :note: the returned array will have F order memory.
    :param src: armadillo object to be converted.
    :param copy: copy the memory of the array, default is false which steals the memory

.. function:: py::array_t<T> Cube_to_arr(const arma::Cube<T>& src)

    Copy the memory of the Armadillo Cube in the conversion to Numpy array.
    
    :note: the returned array will have F order memory.
    :param src: armadillo object to be converted.

.. function:: py::array_t<T> cube_to_arr(arma::Cube<T>&& src)

    Steal the memory of the Armadillo cube in the conversion to Numpy array.
    
    :note: the returned array will have F order memory.
    :param src: armadillo object to be converted.

.. function:: py::array_t<T> cube_to_arr(arma::Cube<T>* src, bool copy=false)

    Convert Armadillo matrix to Numpy array.
    
    :note: the returned array will have F order memory.
    :param src: armadillo object to be converted.
    :param copy: copy the memory of the array, default is false

to_numpy
--------
    
``to_numpy`` has overloads for ``Mat<T>``, ``Row<T>``, ``Col<T>`` and ``Cube<T>``.
It should be called with e.g. ``to_numpy<arma::Mat<double>>(m)``

.. function:: template <typename armaT> py::array_t<eT> to_numpy_view(const armaT<eT>& src)

   Create 'view' on Armadillo object as non-writeable Numpy array.
   :note: the returned array will have F order memory.
   :param src: armadillo object to be converted.

.. function:: template <typename armaT> py::array_t<eT> to_numpy(armaT<eT>& src, bool copy=false)

   Convert Armadillo object to Numpy array.

   :note: the returned array will have F order memory.
   :param src: armadillo object to be converted.
   :param copy: copy the memory of the array, default is false which steals the memory

.. function:: template <typename armaT> py::array_t<eT> to_numpy(const armaT<eT>& src)

   Copy the memory of the Armadillo object in the conversion to Numpy array.

   :note: the returned array will have F order memory.
   :param src: armadillo object to be converted.

.. function:: template <typename armaT> py::array_t<eT> to_numpy(armaT<eT>&& src)

   Steal the memory of the Armadillo object in the conversion to Numpy array.

   :note: the returned array will have F order memory.
   :param src: armadillo object to be converted.

.. function:: template <typename armaT> py::array_t<eT> to_numpy(armaT<eT>* src)

   Convert Armadillo object to Numpy array.

   :note: the returned array will have F order memory.
   :param src: armadillo object to be converted.
   :param copy: copy the memory of the array, default is false

###################################
Automatic conversion -- Type caster
###################################

CARMA provides a type caster which enables automatic conversion using pybind11.

.. warning:: `carma` should included in every compilation unit where automated type casting occurs, otherwise undefined behaviour will occur.

.. note:: The underlying casting function has overloads for ``{const lvalue, lvalue, rvalue, pointer}`` Armadillo objects of type ``{Mat, Col, Row, Cube}``.

.. _return_policies:

Return policies
***************

Pybind11 provides a number of return value policies of which a subset is supported:

To pass the return value policy set it in the binding function:

.. code-block:: c++

    m.def("example_function", &example_function, return_value_policy::copy);

.. function:: return_value_policy::move
    
    move/steal the memory from the armadillo object

.. function:: return_value_policy::automatic

    move/steal the memory from the armadillo object

.. function:: return_value_policy::take_ownership

    move/steal the memory from the armadillo object

.. function:: return_value_policy::copy

    copy the memory from the armadillo object

##########
ArrayStore
##########

ArrayStore is a convenience class for storing the memory in a C++ class.

.. warning:: 
    The ArrayStore owns the data, the returned numpy arrays are views that
    are tied to the lifetime of ArrayStore.

.. class:: template <typename armaT> ArrayStore

       .. attribute:: mat armaT
           
           Matrix containing the memory of the array.

       .. method:: ArrayStore(py::array_t<T>& arr, bool copy)

           Class constructor

           :param arr: Numpy array to be stored as Armadillo matrix
           :param steal: Take ownership of the array if not copy

       .. method:: template <typename armaT> ArrayStore(const arma & src)

           Class constructor, object is copied

           :param src: Armadillo object to be stored

       .. method:: template <typename armaT> ArrayStore(armaT& src, copy)

           Class constructor, object is copied or moved/stolen

           :param src: Armadillo object to be stored

       .. method:: template <typename armaT> ArrayStore(armaT && src)

           Class constructor, object is moved

           :param mat: Armadillo object to be stored

       .. method:: get_view(bool writeable)

           Obtain a view of the memory as Numpy array.

           :param writeable: Mark array as writeable

       .. method:: set_array(py::array_t<T> & arr, bool copy)

           Store new array in the ArrayStore.

           :param arr: Numpy array to be stored as Armadillo matrix
           :param copy: Take ownership of the array or copy

       .. method:: template <typename T> set_data(const armaT& src)

           Store new matrix in the ArrayStore, object is copied.

           :param src: Armadillo object to be stored

       .. method:: template <typename T> set_data(armaT& src, bool copy)

           Store new object in the ArrayStore, copied or moved.

           :param src: Armadillo object to be stored, matrix is copied

       .. method:: template <typename T> set_data(arma::Mat<T> && src)

           Store new matrix in the ArrayStore, object is moved.

           :param src: Armadillo matrix to be stored

#############
NdArray flags
#############

Utility functions to check flags of numpy arrays.

.. function:: bool is_f_contiguous(const py::array_t<T> & arr)

   Check if Numpy array's  memory is Fotran contiguous.

   :param arr: numpy array to be checked

.. function:: bool is_c_contiguous(const py::array_t<T> & arr)

   Check if Numpy array's  memory is C contiguous.

   :param arr: numpy array to be checked

.. function:: bool is_writable(const py::array_t<T> & arr)

   Check if Numpy array's memory is mutable.

   :param arr: numpy array to be checked

.. function:: bool is_owndata(const py::array_t<T> & arr)

   Check if Numpy array's memory is owned by numpy.

   :param arr: numpy array to be checked

.. function:: bool is_aligned(const py::array_t<T> & arr)

   Check if Numpy array's memory is aligned.

   :param arr: numpy array to be checked

.. function:: bool requires_copy(const py::array_t<T> & arr)

   Check if Numpy array memory needs to be copied out, is true
   when either not writable, owndata or is not aligned.

   :param arr: numpy array to be checked

.. function:: void set_not_owndata(py::array_t<T> & arr)

   Set Numpy array's flag OWNDATA to false.

   :param arr: numpy array to be changed

.. function:: void set_not_writeable(py::array_t<T> & arr)

   Set Numpy array's flag WRITEABLE to false.

   :param arr: numpy array to be changed
