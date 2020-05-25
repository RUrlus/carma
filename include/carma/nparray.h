/*  carma/nparray.h: Condition checks numpy arrays
 *  Copyright (c) 2020 Ralph Urlus <rurlus.dev@gmail.com>
 *  All rights reserved. Use of this source code is governed by a
 *  Apache-2.0 license that can be found in the LICENSE file.
 *
 *  Adapated from:
 *      pybind11/numpy.h: Basic NumPy support, vectorize() wrapper
 *
 *      Copyright (c) 2016 Wenzel Jakob <wenzel.jakob@epfl.ch>
 *      All rights reserved. Use of this source code is governed by a
 *      BSD-style license that can be found in the LICENSE file.
*/

/* External headers */
#include <memory>
#include <utility>
#include <type_traits>

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
        return is_f_contiguous(arr) || is_c_contiguous(arr);
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
        #ifdef CARMA_DONT_REQUIRE_OWNDATA
        return (!is_writable(arr) || !is_aligned(arr));
        #else
        return (!is_writable(arr) || !is_owndata(arr) || !is_aligned(arr));
        #endif
    }

} /* carma */

#endif /* NPARRAY */
