/*  carma/numpytoarma.h: Coverter of Numpy arrays to Armadillo matrices
 *  Copyright (c) 2020 Ralph Urlus <rurlus.dev@gmail.com>
 *  All rights reserved. Use of this source code is governed by a
 *  Apache-2.0 license that can be found in the LICENSE file.
 *
 *  Adapated from:
 *
 *      pybind11/eigen.h: Transparent conversion for dense and sparse Eigen matrices
 *      Copyright (c) 2016 Wolf Vollprecht <w.vollprecht@gmail.com>
 *                         Wenzel Jakob <wenzel.jakob@epfl.ch>
 *      All rights reserved. Use of this source code is governed by a
 *      BSD-style license that can be found in the pybind11/LICENSE file.
 *
 *      arma_wrapper/arma_wrapper.h:
 *      Copyright (C) 2019 Paul Sangrey governed by Apache 2.0 License
 */

/* External headers */
#include <armadillo>  // NOLINT
#include <pybind11/buffer_info.h>  // NOLINT
#include <pybind11/detail/common.h>  // NOLINT
#include <pybind11/numpy.h>  // NOLINT
#include <pybind11/pybind11.h>  // NOLINT

/* carma headers */
#include <carma/carma/utils.h>  // NOLINT
#include <carma/carma/cnumpy.h>  // NOLINT
#include <carma/carma/nparray.h>  // NOLINT

namespace py = pybind11;

#ifndef INCLUDE_CARMA_CARMA_NUMPYTOARMA_H_
#define INCLUDE_CARMA_CARMA_NUMPYTOARMA_H_

namespace carma {

using uword = arma::uword;
using aconf =  arma::arma_config;

struct conversion_error : std::exception {
    const char* _message;
    explicit conversion_error(const char* message) : _message(message) {}
    const char* what() const throw() { return _message; }
};

template <typename T>
inline T* _validate_from_array_mat(py::buffer_info& src) {
    T* data = reinterpret_cast<T*>(src.ptr);
    ssize_t dims = src.ndim;
    if (dims < 1 || dims > 2) {
        throw conversion_error("Number of dimensions must be 1 <= ndim <= 2");
    }
    if (data == nullptr) {
        throw conversion_error("CARMA: Array doesn't hold any data, nullptr");
    }
    return data;
}  // _validate_to_array_mat

template <typename T>
inline arma::Mat<T> _arr_to_mat(
    py::buffer_info& src, T* data, bool stolen, bool strict
) {
    // extract buffer information
    ssize_t dims = src.ndim;
    uword nrows;
    uword ncols;
    uword nelem = src.size;

    if (src.ndim == 1) {
        nrows = nelem;
        ncols = 1;
    } else {
        nrows = src.shape[0];
        ncols = src.shape[1];
    }

    /* Handling small arrays
     *
     * ARMA assumes that it's objects with less than mat_prealloc have
     * been stack allocated. Hence, the memory will not be free'd in
     * case of construction.
     *
     * Since the data is soo small copying is not a big deal,
     * we free the array after if it was stolen as arma will
     * not own it.
     */
    bool copy = (nelem > aconf::mat_prealloc) ? false : true;

    arma::Mat<T> dest(data, nrows, ncols, copy, strict);

    if (!stolen) {
        return dest;
    }
    if (!copy) {
        // after stealing Arma has to manage the lifetime of the memory
        arma::access::rw(dest.n_alloc) = nelem;
        arma::access::rw(dest.mem_state) = 0;
        return dest;
    }
    free(data);
    return dest;
} /* _arr_to_mat */

template <typename T>
inline T* _validate_from_array_col(py::buffer_info& src) {
    T* data = reinterpret_cast<T*>(src.ptr);
    ssize_t dims = src.ndim;
    if ((dims >= 2) && (src.shape[1] != 1)) {
        throw conversion_error("Number of columns must <= 1");
    }
    if (src.ptr == nullptr) {
        throw conversion_error("CARMA: Array doesn't hold any data, nullptr");
    }
    return data;
}  // _validate_to_array_col

template <typename T>
arma::Col<T> _arr_to_col(
    py::buffer_info& src, T* data, bool stolen, bool strict
) {
    // extract buffer information
    ssize_t dims = src.ndim;
    uword nelem = src.size;

    bool copy = (nelem > aconf::mat_prealloc) ? false : true;
    arma::Col<T> dest(data, nelem, copy, strict);
    if (!stolen) {
        return dest;
    }
    if (!copy) {
        // after stealing Arma has to manage the lifetime of the memory
        arma::access::rw(dest.n_alloc) = nelem;
        arma::access::rw(dest.mem_state) = 0;
        return dest;
    }
    free(data);
    return dest;
} /* _arr_to_col */

template <typename T>
inline T* _validate_from_array_row(py::buffer_info& src) {
    T* data = reinterpret_cast<T*>(src.ptr);
    ssize_t dims = src.ndim;
    if ((dims >= 2) && (src.shape[0] != 1)) {
        throw conversion_error("Number of rows must <= 1");
    }

    if (src.ptr == nullptr) {
        throw conversion_error("armadillo matrix conversion failed, nullptr");
    }
    return data;
}  // _validate_to_array_row

template <typename T>
arma::Row<T> _arr_to_row(
    py::buffer_info& src, T* data, bool stolen, bool strict
) {
    // extract buffer information
    ssize_t dims = src.ndim;
    uword nelem = src.size;

    bool copy = (nelem > aconf::mat_prealloc) ? false : true;
    arma::Row<T> dest(data, nelem, copy, strict);
    if (!stolen) {
        return dest;
    }
    if (!copy) {
        // after stealing Arma has to manage the lifetime of the memory
        arma::access::rw(dest.n_alloc) = nelem;
        arma::access::rw(dest.mem_state) = 0;
        return dest;
    }
    free(data);
    return dest;
} /* _arr_to_Row */

template <typename T>
inline T* _validate_from_array_cube(py::buffer_info& src) {
    T* data = reinterpret_cast<T*>(src.ptr);
    ssize_t dims = src.ndim;
    if (dims != 3) {
        throw conversion_error("Number of dimensions must be 3");
    }
    if (src.ptr == nullptr) {
        throw conversion_error("CARMA: Array doesn't hold any data, nullptr");
    }
    return data;
}  // _validate_to_array_cube

template <typename T>
arma::Cube<T> _arr_to_cube(
    py::buffer_info& src, T* data, bool stolen, bool strict
) {

    // extract buffer information
    ssize_t dims = src.ndim;
    uword nrows = src.shape[0];
    uword ncols = src.shape[1];
    uword nslices = src.shape[2];
    uword nelem = src.size;

    bool copy = (nelem > arma::Cube_prealloc::mem_n_elem) ? false : true;
    arma::Cube<T> dest(data, nrows, ncols, nslices, copy, strict);
    if (!stolen) {
        return dest;
    }
    if (!copy) {
        // after stealing Arma has to manage the lifetime of the memory
        arma::access::rw(dest.n_alloc) = nelem;
        arma::access::rw(dest.mem_state) = 0;
        return dest;
    }
    free(data);
    return dest;
} /* _arr_to_cube */

}  // namespace carma

#endif  // INCLUDE_CARMA_CARMA_NUMPYTOARMA_H_
