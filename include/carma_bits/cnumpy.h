/*  carma/cnumpy.h: Code to steal the memory from Numpy arrays
 *  Copyright (c) 2020 Ralph Urlus <rurlus.dev@gmail.com>
 *  All rights reserved. Use of this source code is governed by a
 *  Apache-2.0 license that can be found in the LICENSE file.
 */

#ifndef INCLUDE_CARMA_BITS_CNUMPY_H_
#define INCLUDE_CARMA_BITS_CNUMPY_H_
#define NPY_NO_DEPRECATED_API NPY_1_14_API_VERSION
/* C headers */
#include <Python.h>
#include <pymem.h>
#include <numpy/arrayobject.h>
#include <numpy/ndarraytypes.h>

/* STD header */
#include <limits>

/* External */
#include <pybind11/numpy.h>  // NOLINT
#include <pybind11/pybind11.h>  // NOLINT

#include <carma_bits/numpyapi.h> // NOLINT

#include <armadillo> // NOLINT

namespace py = pybind11;

extern "C" {
static inline void steal_memory(PyObject* src) {
    /* ---- steal_memory ----
     * The default behaviour is to replace the stolen array with an array containing
     * a single NaN and set the appropriate dimensions and strides.
     * This means the original references can be accessed but no longer should.
     *
     * Alternative is to define CARMA_HARD_STEAL which sets a nullptr and decreases
     * the reference count. NOTE, accessing the original reference when using
     * CARMA_HARD_STEAL will trigger a segfault.
     *
     * Note this function makes use of PyArrayObject_fields which is internal
     * and is noted with:
     *
     * "The main array object structure. It has been recommended to use the inline
     * functions defined below (PyArray_DATA and friends) to access fields here
     * for a number of releases. Direct access to the members themselves is
     * deprecated. To ensure that your code does not use deprecated access,
     * #define NPY_NO_DEPRECATED_API NPY_1_7_API_VERSION (or
     * NPY_1_8_API_VERSION or higher as required).
     * This struct will be moved to a private header in a future release"
     *
     * I don't know of a way around it, assignment using the macros,
     * such as PyArray_DATA, is not possible.
     * The deprecation macro has been included which will at least raise an error
     * during compilation if `PyArrayObject_fields` has become deprecated.
     *
     * Better solutions are very welcome.
     */
#ifdef CARMA_HARD_STEAL
    reinterpret_cast<PyArrayObject_fields *>(src)->data = nullptr;
#else
    PyArrayObject_fields* obj = reinterpret_cast<PyArrayObject_fields *>(src);
    double* data = reinterpret_cast<double *>(carman::npy_api::get().PyDataMem_NEW_(sizeof(double)));
    data[0] = NAN;
    obj->data = reinterpret_cast<char*>(data);

    size_t ndim = obj->nd;
    obj->nd = 1;
    if (ndim == 1) {
        obj->dimensions[0] = static_cast<npy_int>(1);
    } else if (ndim == 2) {
        obj->dimensions[0] = static_cast<npy_int>(1);
        obj->dimensions[1] = static_cast<npy_int>(0);
    } else {
        obj->dimensions[0] = static_cast<npy_int>(1);
        obj->dimensions[1] = static_cast<npy_int>(0);
        obj->dimensions[2] = static_cast<npy_int>(0);
    }
#endif
}  // steal_memory

}  // extern "C"

namespace carma {

/* Use Numpy's api to account for stride, order and steal the memory */
template <typename T>
inline static T* steal_copy_array(PyObject* src0) {
    PyArrayObject* src = reinterpret_cast<PyArrayObject*>(src0);
    auto& api = carman::npy_api::get();

#if WIN32
    // must be false for WIN32 (cf https://devblogs.microsoft.com/oldnewthing/20060915-04/?p=29723)
    const bool allow_foreign_allocator = false;
#else /* WIN32 */
    const bool allow_foreign_allocator = true;
#endif
    PyArray_Descr* dtype = PyArray_DESCR(src);
    Py_INCREF(dtype);
    int ndim = PyArray_NDIM(reinterpret_cast<PyArrayObject*>(src));
    npy_intp const* dims = PyArray_DIMS(src);

    T* data = NULL;
    npy_intp* strides = NULL;
    const int subok = 1;

    bool delegate_allocation = PyArray_FLAGS(src) & NPY_ARRAY_F_CONTIGUOUS && allow_foreign_allocator;

    if (!delegate_allocation) {
        // we allocate a new memory buffer
        int buffsize = 1;
        for (int d = 0; d < ndim; ++d)
            buffsize *= dims[d];
        data = arma::memory::acquire<T>(buffsize); // data will be freed by arma::memory::release<T>

        strides = new npy_intp[NPY_MAXDIMS];
        npy_intp stride = dtype->elsize;
        for (int idim = 0; idim < ndim; ++idim) {
            strides[idim] = stride;
            stride *= dims[idim];
        }
    }

    // build an PyArray to do F-order copy
    PyArrayObject* dest = reinterpret_cast<PyArrayObject*>(api.PyArray_NewFromDescr_(
        subok ? Py_TYPE(src) : &PyArray_Type,
        dtype,
        ndim,
        dims,
        strides,
        data,
        NPY_FORTRANORDER | ((data) ? ~NPY_ARRAY_OWNDATA : 0),  // | NPY_ARRAY_F_CONTIGUOUS /* | NPY_ARRAY_WRITEABLE*/,
        subok ? reinterpret_cast<PyObject*>(src) : NULL));

    // copy the array to a well behaved F-order
    api.PyArray_CopyInto_(dest, src);

    if (delegate_allocation) {
        // we steal the memory
        PyArrayObject_fields* arr = reinterpret_cast<PyArrayObject_fields*>(dest);
        data = reinterpret_cast<T*>(std::exchange(arr->data, nullptr));
    }
    // free the array
    api.PyArray_Free_(dest, static_cast<void*>(nullptr));
    delete[] strides;

    return data;
}  // steal_copy_array

}  // namespace carma

#endif  // INCLUDE_CARMA_BITS_CNUMPY_H_
