/*  carma/cnumpy.h: Code to steal the memory from Numpy arrays
 *  Copyright (c) 2020 Ralph Urlus <rurlus.dev@gmail.com>
 *  All rights reserved. Use of this source code is governed by a
 *  Apache-2.0 license that can be found in the LICENSE file.
 */

#ifndef INCLUDE_CARMA_CARMA_CNUMPY_H_
#define INCLUDE_CARMA_CARMA_CNUMPY_H_
#define NPY_NO_DEPRECATED_API NPY_1_14_API_VERSION
/* C headers */
#include <Python.h>
#include <numpy/arrayobject.h>
#include <numpy/ndarraytypes.h>

/* STD header */
#include <limits>

/* External */
#include <pybind11/numpy.h>  // NOLINT
#include <pybind11/pybind11.h>  // NOLINT

#include <iostream>

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
    double* data = reinterpret_cast<double *>(malloc(sizeof(double)));
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

/* Use Numpy's api to copy, accounting miss behaved memory, and steal the memory */
static inline void* c_steal_copy_array(PyObject* src) {
    // this is needed as otherwise we get mysterious segfaults
    import_array();
    // copy the array to a well behaved F-order
    PyObject* dest = PyArray_NewCopy(reinterpret_cast<PyArrayObject *>(src), NPY_FORTRANORDER);
    // we steal the memory
    PyArrayObject* arr = reinterpret_cast<PyArrayObject *>(dest);
    void* data = PyArray_DATA(arr);
    reinterpret_cast<PyArrayObject_fields *>(dest)->data = nullptr;
    // free the array
    PyArray_Free(dest, nullptr);
    return data;
}  // c_steal_copy_array

/* Copy to fortran and return data ptr
 * We us Numpy's api to account for stride, order of the memory */
static inline PyObject* copy_well_behaved(PyObject* src) {
    // this is needed as otherwise we get mysterious segfaults
    import_array();
    return PyArray_NewCopy(reinterpret_cast<PyArrayObject *>(src), NPY_FORTRANORDER);
}  // c_copy_well_behaved

/* get data pointer from PyObject, steals a reference */
static inline void * c_get_ptr(PyObject* obj) {
    PyArrayObject* arr = reinterpret_cast<PyArrayObject *>(obj);
    void * data = PyArray_DATA(arr);
    PyArray_XDECREF(arr);
    return data;
}  // c_get_ptr

}  // extern "C"

namespace carma {

/* Use Numpy's api to account for stride, order and steal the memory */
template <typename T>
inline T* steal_copy_array(PyObject* src) {
    return reinterpret_cast<T*>(c_steal_copy_array(src));
}  // steal_copy_array

/* get data pointer from PyObject, steals a reference */
template <typename T>
inline T* get_ptr(PyObject* obj) {
    return reinterpret_cast<T*>(c_get_ptr(obj));
}  // steal_copy_array

}  // namespace carma

#endif  // INCLUDE_CARMA_CARMA_CNUMPY_H_
