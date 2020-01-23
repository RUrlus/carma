/*
Adapated from:

    pybind11/eigen.h: Transparent conversion for dense and sparse Eigen matrices
    Copyright (c) 2016 Wolf Vollprecht <w.vollprecht@gmail.com>
                       Wenzel Jakob <wenzel.jakob@epfl.ch>
    All rights reserved. Use of this source code is governed by a
    BSD-style license that can be found in the LICENSE file.
*/

/* External headers */
#include <pybind11/pybind11.h>
#include <pybind11/numpy.h>
namespace py = pybind11;

#ifndef NPARRAY
#define NPARRAY

namespace carma {

    template <typename T> inline bool is_f_contiguous(const py::array_t<T> & arr) {
        return py::detail::check_flags(arr.ptr(), py::detail::npy_api::NPY_ARRAY_F_CONTIGUOUS_);
    }

    template <typename T> inline bool is_c_contiguous(const py::array_t<T> & arr) {
        return py::detail::check_flags(arr.ptr(), py::detail::npy_api::NPY_ARRAY_C_CONTIGUOUS_);
    }

    template <typename T> inline bool is_contiguous(const py::array_t<T> & arr) {
        return is_f_contiguous(arr) || is_contiguous(arr);
    }

    template <typename T> inline bool is_writable(const py::array_t<T> & arr) {
        return py::detail::check_flags(arr.ptr(), py::detail::npy_api::NPY_ARRAY_WRITEABLE_);
    }

    template <typename T> inline bool is_owndata(const py::array_t<T> & arr) {
        return py::detail::check_flags(arr.ptr(), py::detail::npy_api::NPY_ARRAY_OWNDATA_);
    }

    template <typename T> inline bool is_aligned(const py::array_t<T> & arr) {
        return py::detail::check_flags(arr.ptr(), py::detail::npy_api::NPY_ARRAY_ALIGNED_);
    }

    template <typename T> inline bool requires_copy(const py::array_t<T> & arr) {
        return (!is_writable(arr) || !is_owndata(arr) || !is_aligned(arr));
    }

} /* carma */

#endif /* NPARRAY */
