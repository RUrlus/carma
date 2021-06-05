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

#include <iostream>
/* External headers */
#include <armadillo>  // NOLINT
#include <pybind11/buffer_info.h>  // NOLINT
#include <pybind11/detail/common.h>  // NOLINT
#include <pybind11/numpy.h>  // NOLINT
#include <pybind11/pybind11.h>  // NOLINT

/* carma headers */
#include <carma_bits/debug.h>  // NOLINT
#include <carma_bits/typecheck.h>  // NOLINT
#include <carma_bits/cnumpy.h>  // NOLINT
#include <carma_bits/nparray.h>  // NOLINT
#include <carma_bits/config.h> // NOLINT

namespace py = pybind11;

#ifndef INCLUDE_CARMA_BITS_NUMPYTOARMA_H_
#define INCLUDE_CARMA_BITS_NUMPYTOARMA_H_

namespace carma {

struct conversion_error : std::exception {
    const char* message;
    explicit conversion_error(const char* message) : message(message) {}
    const char* what() const throw() { return message; }
};

namespace details {

using uword = arma::uword;
using aconf =  arma::arma_config;

template<typename T> inline void free_array(T* data) {
#ifdef CARMA_EXTRA_DEBUG
    debug::print_opening();
    std::cout << "Freeing memory @" << data << " of stolen array\n";
    debug::print_closing();
#endif
    arma::memory::release<T>(data);
}  // free_array

template <typename T>
    inline T* steal_andor_copy(PyObject* obj, T* data) {
    if (!well_behaved(obj)) {
#ifdef CARMA_EXTRA_DEBUG
        debug::print_copy_of_data(data);
#endif
        // copy and ensure fortran order
        data = steal_copy_array<T>(obj);
    } else {
        // remove control of memory from numpy
        steal_memory<T>(obj);
    }
    return data;
}


template <typename T>
inline T* validate_from_array_mat(const py::buffer_info& src) {
    T* data = reinterpret_cast<T*>(src.ptr);
    ssize_t dims = src.ndim;
    if (dims < 1 || dims > 2) {
        throw conversion_error("Number of dimensions must be 1 <= ndim <= 2");
    }
    if (data == nullptr) {
        throw conversion_error("CARMA: Array doesn't hold any data, nullptr");
    }
    return data;
}  // validate_to_array_mat

template <typename T>
inline arma::Mat<T> arr_to_mat(
    const py::buffer_info& src, T* data, bool stolen, bool strict
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
#ifdef CARMA_EXTRA_DEBUG
    if (copy) {
        debug::print_prealloc<T>(data);
    }
#endif
// rvalue return due to lack of NRVO on MSVC
#ifdef _WIN32
    // not stolen means numpy owns the memory and Arma borrows the memory
    if (!stolen) {
        return arma::Mat<T> (data, nrows, ncols, copy, strict);
    }
    arma::Mat<T> dest(data, nrows, ncols, copy, strict);
#else
    arma::Mat<T> dest(data, nrows, ncols, copy, strict);
    if (!stolen) {
        // rvalue return due to lack of NRVO on MSVC
        return dest;
    }
#endif
    // we have stolen, numpy no longer owns the memory.
    // but we have copied into the matrix, hence we have to free the memory
    if (copy) {
        free_array(data);
        return dest;
    }
    // we have stolen, numpy no longer owns the memory and
    // we haven't copied into the matrix hence Arma has to manage the lifetime
    // of the memory
    arma::access::rw(dest.n_alloc) = nelem;
    arma::access::rw(dest.mem_state) = 0;
    return dest;
} /* arr_to_mat */

template <typename T>
inline T* validate_from_array_col(const py::buffer_info& src) {
    T* data = reinterpret_cast<T*>(src.ptr);
    ssize_t dims = src.ndim;
    if ((dims >= 2) && (src.shape[1] != 1)) {
        throw conversion_error("CARMA: Number of columns must <= 1");
    }
    if (src.ptr == nullptr) {
        throw conversion_error("CARMA: Array doesn't hold any data, nullptr");
    }
    return data;
}  // validate_to_array_col

template <typename T>
inline arma::Col<T> arr_to_col(
    const py::buffer_info& src, T* data, bool stolen, bool strict
) {
    // extract buffer information
    ssize_t dims = src.ndim;
    uword nelem = src.size;

    bool copy = (nelem > aconf::mat_prealloc) ? false : true;
#ifdef CARMA_EXTRA_DEBUG
    if (copy) {
        debug::print_prealloc<T>(data);
    }
#endif
// rvalue return due to lack of NRVO on MSVC
#ifdef _WIN32
    // not stolen means numpy owns the memory and Arma borrows the memory
    if (!stolen) {
        return arma::Col<T>(data, nelem, copy, strict);
    }
    arma::Col<T> dest(data, nelem, copy, strict);
#else
    arma::Col<T> dest(data, nelem, copy, strict);
    if (!stolen) {
        // rvalue return due to lack of NRVO on MSVC
        return dest;
    }
#endif
    // we have stolen, numpy no longer owns the memory.
    // but we have copied into the matrix, hence we have to free the memory
    if (copy) {
        free_array(data);
        return dest;
    }
    // we have stolen, numpy no longer owns the memory and
    // we haven't copied into the matrix hence Arma has to manage the lifetime
    // of the memory
    arma::access::rw(dest.n_alloc) = nelem;
    arma::access::rw(dest.mem_state) = 0;
    return dest;
} /* arr_to_col */

template <typename T>
inline T* validate_from_array_row(const py::buffer_info& src) {
    T* data = reinterpret_cast<T*>(src.ptr);
    ssize_t dims = src.ndim;
    if ((dims >= 2) && (src.shape[0] != 1)) {
        throw conversion_error("CARMA: Number of rows must <= 1");
    }

    if (src.ptr == nullptr) {
        throw conversion_error("CARMA: armadillo matrix conversion failed, nullptr");
    }
    return data;
}  // validate_to_array_row

template <typename T>
inline arma::Row<T> arr_to_row(
    const py::buffer_info& src, T* data, bool stolen, bool strict
) {
    // extract buffer information
    ssize_t dims = src.ndim;
    uword nelem = src.size;

    bool copy = (nelem > aconf::mat_prealloc) ? false : true;
#ifdef CARMA_EXTRA_DEBUG
    if (copy) {
        debug::print_prealloc<T>(data);
    }
#endif
// rvalue return due to lack of NRVO on MSVC
#ifdef _WIN32
    // not stolen means numpy owns the memory and Arma borrows the memory
    if (!stolen) {
        // rvalue return due to lack of NRVO on MSVC
        return arma::Row<T>(data, nelem, copy, strict);
    }
    arma::Row<T> dest(data, nelem, copy, strict);
#else
    arma::Row<T> dest(data, nelem, copy, strict);
    // not stolen means numpy owns the memory and Arma borrows the memory
    if (!stolen) {
        // rvalue return due to lack of NRVO on MSVC
        return dest;
    }
#endif
    // we have stolen, numpy no longer owns the memory.
    // but we have copied into the matrix, hence we have to free the memory
    if (copy) {
        free_array(data);
        return dest;
    }
    // we have stolen, numpy no longer owns the memory and
    // we haven't copied into the matrix hence Arma has to manage the lifetime
    // of the memory
    arma::access::rw(dest.n_alloc) = nelem;
    arma::access::rw(dest.mem_state) = 0;
    return dest;
} /* arr_to_Row */

template <typename T>
inline T* validate_from_array_cube(const py::buffer_info& src) {
    T* data = reinterpret_cast<T*>(src.ptr);
    ssize_t dims = src.ndim;
    if (dims != 3) {
        throw conversion_error("CARMA: Number of dimensions must be 3");
    }
    if (src.ptr == nullptr) {
        throw conversion_error("CARMA: Array doesn't hold any data, nullptr");
    }
    return data;
}  // validate_to_array_cube

template <typename T>
inline arma::Cube<T> arr_to_cube(
    const py::buffer_info& src, T* data, bool stolen, bool strict
) {
    // extract buffer information
    ssize_t dims = src.ndim;
    uword nrows = src.shape[0];
    uword ncols = src.shape[1];
    uword nslices = src.shape[2];
    uword nelem = src.size;

    bool copy = (nelem > arma::Cube_prealloc::mem_n_elem) ? false : true;
#ifdef CARMA_EXTRA_DEBUG
    if (copy) {
        debug::print_prealloc<T>(data);
    }
#endif
// rvalue return due to lack of NRVO on MSVC
#ifdef _WIN32
    // not stolen means numpy owns the memory and Arma borrows the memory
    if (!stolen) {
        // rvalue return due to lack of NRVO on MSVC
        return arma::Cube<T>(data, nrows, ncols, nslices, copy, strict);
    }
    arma::Cube<T> dest(data, nrows, ncols, nslices, copy, strict);
#else
    arma::Cube<T> dest(data, nrows, ncols, nslices, copy, strict);
    // not stolen means numpy owns the memory and Arma borrows the memory
    if (!stolen) {
        // rvalue return due to lack of NRVO on MSVC
        return dest;
    }
#endif
    // we have stolen, numpy no longer owns the memory.
    // but we have copied into the matrix, hence we have to free the memory
    if (copy) {
        free_array(data);
        return dest;
    }
    // we have stolen, numpy no longer owns the memory and
    // we haven't copied into the matrix hence Arma has to manage the lifetime
    // of the memory
    arma::access::rw(dest.n_alloc) = nelem;
    arma::access::rw(dest.mem_state) = 0;
    return dest;
} /* arr_to_cube */

}  // namespace details
}  // namespace carma

#endif  // INCLUDE_CARMA_BITS_NUMPYTOARMA_H_
