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

#include<carma/carma/numpyapi.h> // NOLINT

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
    double* data = reinterpret_cast<double *>(carma::api::npy_api::get().PyDataMem_NEW_(sizeof(double)));
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
static inline char* c_steal_copy_array(PyObject* src) {
    auto &api = carma::api::npy_api::get();
    // copy the array to a well behaved F-order
    PyObject* dest = api.PyArray_NewCopy_(src, NPY_FORTRANORDER);
    // we steal the memory
    PyArrayObject_fields* arr = reinterpret_cast<PyArrayObject_fields *>(dest);
    char* data = arr->data;
    arr->data = nullptr;
    // free the array
    api.PyArray_Free_(dest, static_cast<void *>(nullptr));
    return data;
}  // c_steal_copy_array

}  // extern "C"

namespace carma {

/* Use Numpy's api to account for stride, order and steal the memory */
template <typename T>
inline T* steal_copy_array(PyObject* src) {
    return reinterpret_cast<T*>(c_steal_copy_array(src));
}  // steal_copy_array

}  // namespace carma

#endif  // INCLUDE_CARMA_CARMA_CNUMPY_H_
