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
#include <iostream>
#include <limits>

/* External */
#include <pybind11/numpy.h>  // NOLINT
#include <pybind11/pybind11.h>  // NOLINT

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
     * Py_XDECREF #define NPY_NO_DEPRECATED_API NPY_1_7_API_VERSION (or
     * NPY_1_8_API_VERSION or higher as required).
     * This struct will be moved to a private header in a future release"
     *
     * I don't know of a way around it, assignment using the macros,
     * such as PyArray_DATA, is not possible.
     * The deprecation macro has been included which will at least raise an error
     * during compilation if `PyArrayObject_fields` has become deprecated.
     *
     * Better alternative are very welcome.
     */
    // FIXME [RURLUS] -- sort out the refcount
    PyArrayObject_fields* obj = reinterpret_cast<PyArrayObject_fields *>(src);
#ifdef CARMA_HARD_STEAL
    obj->data = nullptr;
    // FIXME
    // Find out how many references we hold on our end for the various paths
    // and decref such that all references to the array are garbage collected
    // decrease the reference of the PyObject that holds the stolen memory
    Py_XDECREF(src);
#else
    char* data = reinterpret_cast<char *>(malloc(sizeof(double)));
    data[0] = std::numeric_limits<double>::quiet_NaN();
    obj->data = data;

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
    // reducte the refcount as we don't need to hold one anymore
    Py_XDECREF(src);

    // npy_intp* dims = reinterpret_cast<npy_intp*>(malloc(sizeof(npy_int) * 2));
    // dims[0] = static_cast<npy_int>(1);
    // dims[1] = static_cast<npy_int>(1);
    // npy_intp* strides = reinterpret_cast<npy_intp*>(malloc(sizeof(npy_int) * 2));
    // strides[0] = static_cast<npy_int>(sizeof(double));
    // strides[1] = static_cast<npy_int>(0);
#endif
}  // steal_memory

static inline void * c_swap_c_to_f_order(PyObject* src) {
    // this is needed as otherwise we get mysterious segfaults
    import_array();
    PyArrayObject* np_src = reinterpret_cast<PyArrayObject *>(src);
    PyArrayObject* tmp = reinterpret_cast<PyArrayObject *>(PyArray_NewCopy(np_src, NPY_FORTRANORDER));
    int state = PyArray_MoveInto(np_src, tmp);
    if (state == 1) {
        return reinterpret_cast<void *>(src);
    }
    return nullptr;
}  // c_swap_c_to_f_order

/* Use Numpy's api to account for stride, order and steal the memory */
static inline void* c_steal_copy_array(PyObject* src) {
    // this is needed as otherwise we get mysterious segfaults
    import_array();
    PyArrayObject* np_src = reinterpret_cast<PyArrayObject *>(src);
    PyObject* dest = PyArray_NewCopy(np_src, NPY_FORTRANORDER);
    void* data = PyArray_DATA(reinterpret_cast<PyArrayObject *>(dest));
    // we steal the memory as we can't control the refcount correctly on conversion back
    steal_memory(dest);
    return data;
}  // c_steal_copy_array

/* Copy to fortran and return data ptr
 * We us Numpy's api to account for stride, order of the memory */
static inline void* c_as_fortran(PyObject* src) {
    // this is needed as otherwise we get mysterious segfaults
    import_array();
    PyArrayObject* np_src = reinterpret_cast<PyArrayObject *>(src);
    PyObject* dest = PyArray_NewCopy(np_src, NPY_FORTRANORDER);
    return PyArray_DATA(reinterpret_cast<PyArrayObject *>(dest));
}  // c_as_fortran

}  // extern "C"

namespace carma {

template <typename T>
inline py::array_t<T> swap_c_to_f_order(py::handle& src) {
    /* Use Numpy's api to account for stride, order and steal the memory
     * This function is to deal with no templates in extern C
     */
    void * obj = c_swap_c_to_f_order(src.ptr());
    if (obj == nullptr) {
        return src;
    }
    return py::array_t<T>(reinterpret_cast<PyObject *>(obj));
}  // steal_copy_array

/* Use Numpy's api to account for stride, order and steal the memory */
template <typename T>
inline T* steal_copy_array(py::handle& src) {
    void * data = c_steal_copy_array(src.ptr());
    return reinterpret_cast<T*>(data);
}  // steal_copy_array

/* Copy to fortran and return data ptr
 * We us Numpy's api to account for stride, order of the memory */
template <typename T>
inline T* as_fortran(py::handle& src) {
    return reinterpret_cast<T*>(c_as_fortran(src.ptr()));
}  // as_fortran

}  // namespace carma

#endif  // INCLUDE_CARMA_CARMA_CNUMPY_H_
