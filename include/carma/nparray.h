/*  carma/utils.h: Utility functions for arma converters
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

    template <typename T>
    class flat_reference {
        /* get flattened, unsafe, unchecked const access to array */
    private:
        const unsigned char *ptr;
        const size_t tsize;

    public:
        // Constructor
        flat_reference(const T *data)
        : ptr{reinterpret_cast<const unsigned char *>(data)}, tsize{sizeof(T)} {}

        flat_reference(const py::array_t<T> &arr)
        : ptr{reinterpret_cast<const unsigned char *>(arr.data())}, tsize{sizeof(T)} {}

        // offset pointer and dereference
        const T& operator[](size_t index) const {
            return *reinterpret_cast<const T *>(ptr + tsize * index);
        }
    };

    template <typename T>
    class mutable_flat_reference {
        /* get flattened, unsafe, unchecked mutable access to array */
    private:
        const unsigned char *ptr;
        const size_t tsize;

    public:
        // Constructor
        mutable_flat_reference(T *data)
        : ptr{reinterpret_cast<unsigned char *>(data)}, tsize{sizeof(T)} {}

        mutable_flat_reference(py::array_t<T> &arr)
        : ptr{reinterpret_cast<unsigned char *>(arr.mutable_data())}, tsize{sizeof(T)} {}

        // offset pointer and return
        T& operator[](size_t index) {
            return const_cast<T &>(*reinterpret_cast<const T *>(ptr + tsize * index));
        }
    };

} /* carma */

#endif /* NPARRAY */
