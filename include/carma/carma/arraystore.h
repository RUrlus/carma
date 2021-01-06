/*  carma/arraystore.h: Store arrays as armadillo matrices
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
#include <utility>
#include <iostream>
#include <pybind11/numpy.h>  // NOLINT
#include <pybind11/pybind11.h>  // NOLINT
#include <carma/carma/converters.h>  // NOLINT

namespace py = pybind11;

#ifndef INCLUDE_CARMA_CARMA_ARRAYSTORE_H_
#define INCLUDE_CARMA_CARMA_ARRAYSTORE_H_

namespace carma {

template <typename armaT>
class ArrayStore {
    using T = typename armaT::elem_type;

   protected:
    constexpr static ssize_t tsize = sizeof(T);
    bool _copy;
    py::capsule _base;

   public:
    armaT mat;

   protected:
    inline void _convert_to_arma(py::array_t<T>& arr) {
        mat = _to_arma<armaT>::from(arr, _copy, false);
        _base = create_dummy_capsule(mat.memptr());
        // inform numpy it no longer owns the data
        if (!_copy)
            set_not_owndata(arr);
    }

   public:
    ArrayStore(py::array_t<T>& arr, bool copy) : _copy{copy} {
        /* Constructor
         *
         * Takes numpy array and converters to Armadillo matrix.
         * If the array should be stolen we set owndata false for
         * numpy array.
         *
         * We store a capsule to serve as a reference for the
         * views on the data
         *
         */
        _convert_to_arma(arr);
    }

    explicit ArrayStore(const armaT& src) : _copy{true}, mat{armaT(src)} {
        _base = create_dummy_capsule(mat.memptr());
    }

    ArrayStore(arma::Mat<T>& src, bool copy) : _copy{copy} {
        if (_copy) {
            mat = armaT(src.memptr(), src.n_rows, src.n_cols, true);
        } else {
            mat = std::move(src);
        }
        _base = create_dummy_capsule(mat.memptr());
    }

    ArrayStore(arma::Cube<T>& src, bool copy) : _copy{copy} {
        if (_copy) {
            mat = armaT(src.memptr(), src.n_rows, src.n_cols, src.n_slices, true);
        } else {
            mat = std::move(src);
        }
        _base = create_dummy_capsule(mat.memptr());
    }

    // SFINAE by adding additional parameter as
    // to avoid shadowing the class template
    template <typename U = armaT>
    ArrayStore(armaT& src, bool copy, is_Vec<U>) : _copy{copy} {
        if (_copy) {
            mat = armaT(src.memptr(), src.n_elem, true);
        } else {
            mat = std::move(src);
        }
        _base = create_dummy_capsule(mat.memptr());
    }

    explicit ArrayStore(armaT&& src) noexcept : _copy{false}, mat{std::move(src)} { _base = create_dummy_capsule(mat.memptr()); }

    // Function requires different name than set_data
    // as overload could not be resolved without
    void set_array(py::array_t<T>& arr, bool copy) {
        _copy = copy;
        _convert_to_arma(arr);
    }

    void set_data(const armaT& src) {
        _copy = true;
        mat = armaT(src);
        _base = create_dummy_capsule(mat.memptr());
    }

    void set_data(arma::Mat<T>& src, bool copy) {
        _copy = copy;
        if (copy) {
            mat = armaT(src.memptr(), src.n_rows, src.n_cols, true);
        } else {
            mat = std::move(src);
        }
        _base = create_dummy_capsule(mat.memptr());
    }

    // SFINAE by adding additional parameter as
    // to avoid shadowing the class template
    template <typename U = armaT>
    void set_data(armaT& src, bool copy, is_Vec<U>) {
        _copy = copy;
        if (copy) {
            mat = armaT(src.memptr(), src.n_elem, true);
        } else {
            mat = std::move(src);
        }
        _base = create_dummy_capsule(mat.memptr());
    }

    void set_data(arma::Cube<T>& src, bool copy) {
        _copy = copy;
        if (copy) {
            mat = armaT(src.memptr(), src.n_rows, src.n_cols, src.n_slices, true);
        } else {
            mat = std::move(src);
        }
        _base = create_dummy_capsule(mat.memptr());
    }

    void set_data(armaT&& src) {
        _copy = false;
        mat = std::move(src);
        _base = create_dummy_capsule(mat.memptr());
    }

    py::array_t<T> get_view(bool writeable) {
        ssize_t nslices;
        ssize_t nelem = static_cast<ssize_t>(mat.n_elem);
        ssize_t nrows = static_cast<ssize_t>(mat.n_rows);
        ssize_t ncols = static_cast<ssize_t>(mat.n_cols);
        ssize_t rc_elem = nrows * ncols;

        py::array_t<T> arr;

        // detect cubes
        if (rc_elem != nelem) {
            nslices = nelem / rc_elem;
            arr = py::array_t<T>(
                {nslices, nrows, ncols},                        // shape
                {tsize * nrows * ncols, tsize, nrows * tsize},  // F-style contiguous strides
                mat.memptr(),                                   // the data pointer
                _base                                           // numpy array references this parent
            );
        } else {
            arr = py::array_t<T>(
                {nrows, ncols},          // shape
                {tsize, nrows * tsize},  // F-style contiguous strides
                mat.memptr(),            // the data pointer
                _base                    // numpy array references this parent
            );
        }

        // inform numpy it does not own the buffer
        set_not_owndata(arr);

        if (!writeable)
            set_not_writeable(arr);
        return arr;
    }
};

} /* namespace carma */

#endif  // INCLUDE_CARMA_CARMA_ARRAYSTORE_H_
