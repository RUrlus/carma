/*
Adapated from:

    pybind11/eigen.h: Transparent conversion for dense and sparse Eigen matrices
    Copyright (c) 2016 Wolf Vollprecht <w.vollprecht@gmail.com>
                       Wenzel Jakob <wenzel.jakob@epfl.ch>
    All rights reserved. Use of this source code is governed by a
    BSD-style license that can be found in the pybind11/LICENSE file.

    arma_wrapper/arma_wrapper.h
    Copyright (c) 2019 Paul Sangrey
    https://gitlab.com/sangrey/Arma_Wrapper

    Licensed under the Apache License, Version 2.0 (the "License");
    you may not use this file except in compliance with the License.
    You may obtain a copy of the License at http://www.apache.org/licenses/LICENSE-2.0

*/
#include <memory>
#include <utility>
#include <type_traits>

/* External headers */
#include <armadillo>
#include <pybind11/pybind11.h>
#include <pybind11/numpy.h>
#include <pybind11/buffer_info.h>
#include <pybind11/detail/common.h>
namespace py = pybind11;

/* fasts headers */
#include <carma/utils.h>
#include <carma/nparray.h>

#ifndef ARMA_CONVERTERS
#define ARMA_CONVERTERS

namespace carma {

/*****************************************************************************************
*                                   Numpy to Armadillo                                   *
*****************************************************************************************/
    template <typename T> arma::Mat<T> arr_to_mat(py::handle src, bool copy=false, bool strict=false) {
        /* Convert numpy array to Armadillo Matrix
         *
         * The default behaviour is to avoid copying, we copy if:
         * - ndim == 2 && not F contiguous memory
         * - writable is false
         * - owndata is false
         * - memory is not aligned
         * Note that the user set behaviour is overridden is one of the above conditions
         * is true
         *
         * If the array is 1D we create a column oriented matrix (N, 1)
         */
        // set as array buffer
        py::array_t<T> buffer = py::array_t<T>::ensure(src);
        if (!buffer) {
            throw std::runtime_error("armadillo matrix conversion failed");
        }

        auto dims = buffer.ndim();
        if (dims < 1 || dims > 2) {
            throw std::runtime_error("Number of dimensions must be 1 <= ndim <= 2");
        }

        py::buffer_info info = buffer.request();
        if(info.ptr == nullptr) {
            throw std::runtime_error("armadillo matrix conversion failed, nullptr");
        }

        if (dims == 1) {
            if (requires_copy(buffer)) {
                copy = true;
                strict = false;
            }
            return arma::Mat<T>(static_cast<T *>(info.ptr), buffer.size(), 1, copy, strict);
        }
        if (requires_copy(buffer) || !is_f_contiguous(buffer)) {
            // If not F-contiguous or writable or numpy's data let pybind handle the copy
            buffer = py::array_t<T, py::array::f_style | py::array::forcecast>::ensure(src);
			info = buffer.request();
            copy = false;
            strict = false;
        }
        return arma::Mat<T>(static_cast<T *>(info.ptr), info.shape[0], info.shape[1], copy, strict);
    } /* arr_to_mat */

    template <typename T> arma::Col<T> arr_to_col(py::handle src, bool copy=false, bool strict=false) {
        /* Convert numpy array to Armadillo Column
         *
         * The default behaviour is to avoid copying, we copy if:
         * - writable is false
         * - owndata is false
         * - memory is not aligned
         * Note that the user set behaviour is overridden is one of the above conditions
         * is true
         */
        // set as array buffer
        py::array_t<T> buffer = py::array_t<T>::ensure(src);
        if (!buffer) {
            throw std::runtime_error("armadillo matrix conversion failed");
        }

        auto dims = buffer.ndim();
        if (dims != 1) {
            throw std::runtime_error("Number of dimensions must be 1 <= ndim <= 2");
        }

        py::buffer_info info = buffer.request();
        if(info.ptr == nullptr) {
            throw std::runtime_error("armadillo matrix conversion failed, nullptr");
        }

        if (requires_copy(buffer)) {
            copy = true;
            strict = false;
        }
        return arma::Col<T>(static_cast<T *>(info.ptr), buffer.size(), copy, strict);
    } /* arr_to_col */

    template <typename T> arma::Row<T> arr_to_row(py::handle src, bool copy=false, bool strict=false) {
        /* Convert numpy array to Armadillo Row
         *
         * The default behaviour is to avoid copying, we copy if:
         * - writable is false
         * - owndata is false
         * - memory is not aligned
         * Note that the user set behaviour is overridden is one of the above conditions
         * is true
         */
        // set as array buffer
        py::array_t<T> buffer = py::array_t<T>::ensure(src);
        if (!buffer) {
            throw std::runtime_error("armadillo matrix conversion failed");
        }

        auto dims = buffer.ndim();
        if (dims != 1) {
            throw std::runtime_error("Number of dimensions must be 1 <= ndim <= 2");
        }

        py::buffer_info info = buffer.request();
        if(info.ptr == nullptr) {
            throw std::runtime_error("armadillo matrix conversion failed, nullptr");
        }

        if (requires_copy(buffer)) {
            copy = true;
            strict = false;
        }
        return arma::Row<T>(static_cast<T *>(info.ptr), buffer.size(), copy, strict);
    } /* arr_to_row */

    /* FIXME CREDIT SOURCE
     * This is a templated functor that has overloads that convert the various
     * types that I want to pass from Python to C++.
     */
    template<typename returnT, typename SFINAE=std::true_type> struct _to_arma {
        static_assert(! SFINAE::value, "The general case is not defined.");
        template<typename innerT>
        static returnT from(innerT&&);
    }; /* to_arma */

    template<typename returnT> struct _to_arma<returnT, typename is_row<returnT>::type> {
        /* Overload concept on return type; convert to row */
        template<typename T>
        static returnT from(py::array_t<T>& arr, bool copy, bool strict) {
            return arr_to_row(arr, copy, strict);
        }
    }; /* to_arma */

    template<typename returnT> struct _to_arma<returnT, typename is_col<returnT>::type> {
        /* Overload concept on return type; convert to col */
        template<typename T>
        static returnT from(py::array_t<T>& arr, bool copy, bool strict) {
            return arr_to_col(arr, copy, strict);
        }
    }; /* to_arma */

    template<typename returnT> struct _to_arma<returnT, typename is_mat<returnT>::type> {
        /* Overload concept on return type; convert to matrix */
        template<typename T>
        static returnT from(py::array_t<T>& arr, bool copy, bool strict) {
            return arr_to_mat(arr, copy, strict);
        }
    }; /* to_arma */

    template<typename returnT> struct _to_arma<returnT, typename is_cube<returnT>::type> {
        /* Overload concept on return type; convert to cube */
        template<typename T>
        static returnT from(py::array_t<T>& arr, bool copy, bool strict) {
            return arr_to_cube(arr, copy, strict);
        }
    }; /* to_arma */


/*****************************************************************************************
*                                   Armadillo to Numpy                                   *
*****************************************************************************************/
    template <typename T> py::array_t<T> _row_to_arr(arma::Row<T> && src, bool copy) {
        /* Convert armadillo row to numpy array */
        ssize_t tsize =  static_cast<ssize_t>(sizeof(T));
        ssize_t ncols = static_cast<ssize_t>(src.n_cols);

        T * data = get_data(src, copy);
        py::capsule base = create_capsule(data);

        return py::array_t<T>(
            {1, ncols}, // shape
            {tsize}, // F-style contiguous strides
            data, // the data pointer
            base // numpy array references this parent
        );
    } /* row_to_arr */

    template <typename T> py::array_t<T> row_to_arr(arma::Row<T> && src, bool copy=false) {
        /* Convert armadillo row to numpy array */
        return _row_to_arr(std::forward<arma::Row<T>>(src), copy);
    } /* row_to_arr */

    template <typename T> py::array_t<T> row_to_arr(arma::Row<T> & src, bool copy=false) {
        /* Convert armadillo row to numpy array */
        return _row_to_arr(std::forward<arma::Row<T>>(src), copy);
    } /* row_to_arr */

    template <typename T> py::array_t<T> row_to_arr(arma::Row<T> * src, bool copy=false) {
        /* Convert armadillo row to numpy array */
        return _row_to_arr(std::forward<arma::Row<T>>(* src), copy);
    } /* row_to_arr */

    /* ######################################## Col ######################################## */
    template <typename T> py::array_t<T> _col_to_arr(arma::Col<T> && src, bool copy) {
        /* Convert armadillo col to numpy array */
        ssize_t tsize =  static_cast<ssize_t>(sizeof(T));
        ssize_t nrows = static_cast<ssize_t>(src.n_rows);

        T * data = get_data(src, copy);
        py::capsule base = create_capsule(data);

        return py::array_t<T>(
            {nrows, 1}, // shape
            {tsize}, // F-style contiguous strides
            data, // the data pointer
            base // numpy array references this parent
        );
    } /* _col_to_arr */

    template <typename T> py::array_t<T> col_to_arr(arma::Col<T> && src, bool copy=false) {
        /* Convert armadillo col to numpy array */
        return _col_to_arr(std::forward<arma::Col<T>>(src), copy);
    } /* col_to_arr */

    template <typename T> py::array_t<T> col_to_arr(arma::Col<T> & src, bool copy=false) {
        /* Convert armadillo col to numpy array */
        return _col_to_arr(std::forward<arma::Col<T>>(src), copy);
    } /* col_to_arr */

    template <typename T> py::array_t<T> col_to_arr(arma::Col<T> * src, bool copy=false) {
        /* Convert armadillo col to numpy array */
        return _col_to_arr(std::forward<arma::Col<T>>(* src), copy);
    } /* col_to_arr */

    /* ######################################## Mat ######################################## */
    template <typename T> py::array_t<T> _mat_to_arr(arma::Mat<T> && src, bool copy) {
        /* Convert armadillo matrix to numpy array */
        ssize_t tsize =  static_cast<ssize_t>(sizeof(T));
        ssize_t nrows = static_cast<ssize_t>(src.n_rows);
        ssize_t ncols = static_cast<ssize_t>(src.n_cols);

        T * data = get_data<arma::Mat<T>>(src, copy);
        py::capsule base = create_capsule(data);

        auto arr = py::array_t<T>(
            {nrows, ncols}, // shape
            {tsize, nrows * tsize}, // F-style contiguous strides
            data, // the data pointer
            base // numpy array references this parent
        );
        arr.inc_ref();
        std::cerr << "Mat to arr" << std::endl;
        return arr;
    } /* _mat_to_arr */

    template <typename T> py::array_t<T> mat_to_arr(arma::Mat<T> && src, bool copy=false) {
        return _mat_to_arr(std::forward<arma::Mat<T>>(src), copy);
    } /* mat_to_arr */

    template <typename T> py::array_t<T> mat_to_arr(arma::Mat<T> & src, bool copy=false) {
        return _mat_to_arr(std::forward<arma::Mat<T>>(src), copy);
    } /* mat_to_arr */

    template <typename T> py::array_t<T> mat_to_arr(arma::Mat<T> * src, bool copy=false) {
        return _mat_to_arr(std::forward<arma::Mat<T>>(* src), copy);
    } /* mat_to_arr */

    /* ######################################## Cube ######################################## */
    template <typename T> py::array_t<T> _cube_to_arr(arma::Cube<T> && src, bool copy) {
        /* Convert armadillo matrix to numpy array */
        ssize_t tsize =  static_cast<ssize_t>(sizeof(T));
        ssize_t nrows = static_cast<ssize_t>(src.n_rows);
        ssize_t ncols = static_cast<ssize_t>(src.n_cols);
        ssize_t nslices = static_cast<ssize_t>(src.n_slices);

        T * data = get_data(src, copy);
        py::capsule base = create_capsule(data);

        return py::array_t<T>(
            {nslices, nrows, ncols}, // shape
            {tsize * nrows * ncols, tsize, nrows * tsize}, // F-style contiguous strides
            data, // the data pointer
            base // numpy array references this parent
        );
    } /* _cube_to_arr */

    template <typename T> py::array_t<T> cube_to_arr(arma::Cube<T> && src, bool copy=false) {
        return _cube_to_arr(std::forward<arma::Cube<T>>(src), copy);
    } /* cube_to_arr */

    template <typename T> py::array_t<T> cube_to_arr(arma::Cube<T> & src, bool copy=false) {
        return _cube_to_arr(std::forward<arma::Cube<T>>(src), copy);
    } /* cube_to_arr */

    template <typename T> py::array_t<T> cube_to_arr(arma::Cube<T> * src, bool copy=false) {
        return _cube_to_arr(std::forward<arma::Cube<T>>(* src), copy);
    } /* cube_to_arr */

    /* ---------------------------------- to_numpy ---------------------------------- */
    template <typename T> inline py::array_t<T> to_numpy(arma::Row<T> && src, bool copy) {
        return _row_to_arr(std::forward<arma::Row<T>>(src), copy);
    }

    template <typename T> inline py::array_t<T> to_numpy(arma::Col<T> && src, bool copy) {
        return _col_to_arr(std::forward<arma::Col<T>>(src), copy);
    }

    template <typename T> inline py::array_t<T> to_numpy(arma::Mat<T> && src, bool copy) {
        return _mat_to_arr(std::forward<arma::Mat<T>>(src), copy);
    }

    template <typename T> inline py::array_t<T> to_numpy(arma::Cube<T> && src, bool copy) {
        return _cube_to_arr(std::forward<arma::Cube<T>>(src), copy);
    }


} /* carma */

namespace pybind11 { namespace detail {

template<typename armaT>
struct type_caster<armaT, enable_if_t<carma::is_convertible<armaT>::value>> {
    using T = typename armaT::elem_type;

    /* Convert numpy array to Armadillo Matrix
     *
     * The default behaviour is to avoid copying, we copy if:
     * - ndim == 2 && not F contiguous memory
     * - writable is false
     * - owndata is false
     * - memory is not aligned
     * Note that the user set behaviour is overridden is one of the above conditions
     * is true
     *
     * If the array is 1D we create a column oriented matrix (N, 1) */
    bool load(handle src, bool) {
        // set as array buffer
        bool copy = false;
        bool strict = true;

        py::array_t<T> buffer = py::array_t<T>::ensure(src);
        if (!buffer) {
            return false;
        }

        auto dims = buffer.ndim();
        if (dims < 1 || dims > 3) {
            return false;
        }

        py::buffer_info info = buffer.request();
        if(info.ptr == nullptr) {
            return false;
        }

        value = carma::_to_arma<armaT>::from(info, copy, strict);
        return true;
    }

    private:

        // Cast implementation
        template <typename CType>
        static handle cast_impl(CType && src, return_value_policy policy, handle) {
            switch (policy) {
                case return_value_policy::move:
                    return to_numpy(src, false).release();
                case return_value_policy::automatic:
                    return to_numpy(src, false).release();
                case return_value_policy::take_ownership:
                    return to_numpy(src, false).release();
                case return_value_policy::copy:
                    return to_numpy(src, true).release();
                default:
                    throw cast_error("unhandled return_value_policy");
            };
        }

    public:

        // Normal returned non-reference, non-const value: we steal
        static handle cast(armaT &&src, return_value_policy policy , handle parent) {
            return cast_impl(&src, policy, parent);
        }
        // If you return a non-reference const; we copy
        static handle cast(const armaT &&src, return_value_policy policy /* policy */, handle parent) {
            policy = return_value_policy::copy;
            return cast_impl(std::forward<armaT>(* src), policy, parent);
        }
        // lvalue reference return; default (automatic) becomes steal
        static handle cast(armaT &src, return_value_policy policy, handle parent) {
            return cast_impl(std::forward<armaT>(src), policy, parent);
        }
        // const lvalue reference return; default (automatic) becomes copy
        static handle cast(const armaT &src, return_value_policy policy, handle parent) {
            policy = return_value_policy::copy;
            return cast_impl(std::forward<armaT>(src), policy, parent);
        }
        // non-const pointer return; we steal
        static handle cast(armaT *src, return_value_policy policy, handle parent) {
            return cast_impl(std::forward<armaT>(* src), policy, parent);
        }
        // const pointer return; we copy
        static handle cast(const armaT *src, return_value_policy policy, handle parent) {
            policy = return_value_policy::copy;
            return cast_impl(std::forward<armaT>(* src), policy, parent);
        }

    PYBIND11_TYPE_CASTER(arma::Mat<T>, _("Numpy.ndarray[") + npy_format_descriptor<T>::name() + _("]"));
};
}} // namespace pybind11::detail
#endif /* ARMA_CONVERTERS */
