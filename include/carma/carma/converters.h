/*  carma/carma.h: Coverter of Numpy arrays and Armadillo matrices
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
#include <memory>
#include <type_traits>
#include <utility>

/* External headers */
#include <pybind11/buffer_info.h>
#include <pybind11/detail/common.h>
#include <pybind11/numpy.h>
#include <pybind11/pybind11.h>
#include <armadillo>

/* carma headers */
#include <carma/carma/nparray.h>
#include <carma/carma/utils.h>

namespace py = pybind11;

#ifndef ARMA_CONVERTERS
#define ARMA_CONVERTERS

namespace carma {

/*****************************************************************************************
 *                                   Numpy to Armadillo                                   *
 *****************************************************************************************/
template <typename T>
arma::Mat<T> arr_to_mat(py::handle src, bool copy = false, bool strict = false) {
    /* Convert numpy array to Armadillo Matrix
     *
     * The default behaviour is to avoid copying, we copy if:
     * - ndim == 2 && not F contiguous memory
     * - writeable is false
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
        throw conversion_error("invalid object passed");
    }

    auto dims = buffer.ndim();
    if (dims < 1 || dims > 2) {
        throw conversion_error("Number of dimensions must be 1 <= ndim <= 2");
    }

    py::buffer_info info = buffer.request();
    if (info.ptr == nullptr) {
        throw conversion_error("armadillo matrix conversion failed, nullptr");
    }

    if (dims == 1) {
        if (requires_copy(buffer)) {
            copy = true;
            strict = false;
        }
        return arma::Mat<T>(static_cast<T*>(info.ptr), buffer.size(), 1, copy, strict);
    }

#ifdef CARMA_DONT_REQUIRE_F_CONTIGUOUS
    if (requires_copy(buffer)) {
        copy = false;
        strict = false;
    }
#else
    if (requires_copy(buffer) || !is_f_contiguous(buffer)) {
        // If not F-contiguous or writeable or numpy's data let pybind handle the copy
        buffer = py::array_t<T, py::array::f_style | py::array::forcecast>::ensure(src);
        info = buffer.request();
        copy = false;
        strict = false;
    }
#endif
    return arma::Mat<T>(static_cast<T*>(info.ptr), info.shape[0], info.shape[1], copy, strict);
} /* arr_to_mat */

template <typename T>
arma::Col<T> arr_to_col(py::handle src, bool copy = false, bool strict = false) {
    /* Convert numpy array to Armadillo Column
     *
     * The default behaviour is to avoid copying, we copy if:
     * - writeable is false
     * - owndata is false
     * - memory is not aligned
     * Note that the user set behaviour is overridden is one of the above conditions
     * is true
     */
    // set as array buffer
    py::array_t<T> buffer = py::array_t<T>::ensure(src);
    if (!buffer) {
        throw conversion_error("invalid object passed");
    }

    auto dims = buffer.ndim();
    if ((dims >= 2) && (buffer.shape(1) != 1)) {
        throw conversion_error("Number of columns must <= 1");
    }

    py::buffer_info info = buffer.request();
    if (info.ptr == nullptr) {
        throw conversion_error("armadillo matrix conversion failed, nullptr");
    }

    if (requires_copy(buffer)) {
        copy = true;
        strict = false;
    }
    return arma::Col<T>(static_cast<T*>(info.ptr), buffer.size(), copy, strict);
} /* arr_to_col */

template <typename T>
arma::Row<T> arr_to_row(py::handle src, bool copy = false, bool strict = false) {
    /* Convert numpy array to Armadillo Row
     *
     * The default behaviour is to avoid copying, we copy if:
     * - writeable is false
     * - owndata is false
     * - memory is not aligned
     * Note that the user set behaviour is overridden is one of the above conditions
     * is true
     */
    // set as array buffer
    py::array_t<T> buffer = py::array_t<T>::ensure(src);
    if (!buffer) {
        throw conversion_error("invalid object passed");
    }

    auto dims = buffer.ndim();
    if ((dims >= 2) && (buffer.shape(0) != 1)) {
        throw conversion_error("Number of rows must <= 1");
    }

    py::buffer_info info = buffer.request();
    if (info.ptr == nullptr) {
        throw conversion_error("armadillo matrix conversion failed, nullptr");
    }

    if (requires_copy(buffer)) {
        copy = true;
        strict = false;
    }
    return arma::Row<T>(static_cast<T*>(info.ptr), buffer.size(), copy, strict);
} /* arr_to_row */

template <typename T>
arma::Cube<T> arr_to_cube(py::handle src, bool copy = false, bool strict = false) {
    /* Convert numpy array to Armadillo Cube
     *
     * The default behaviour is to avoid copying, we copy if:
     * - ndim == 2 && not F contiguous memory
     * - writeable is false
     * - owndata is false
     * - memory is not aligned
     * Note that the user set behaviour is overridden is one of the above conditions
     * is true
     *
     */
    // set as array buffer
    py::array_t<T> buffer = py::array_t<T>::ensure(src);
    if (!buffer) {
        throw conversion_error("invalid object passed");
    }

    auto dims = buffer.ndim();
    if (dims != 3) {
        throw conversion_error("Number of dimensions must be 3");
    }

    py::buffer_info info = buffer.request();
    if (info.ptr == nullptr) {
        throw conversion_error("armadillo matrix conversion failed, nullptr");
    }

#ifdef CARMA_DONT_REQUIRE_F_CONTIGUOUS
    if (requires_copy(buffer)) {
        copy = false;
        strict = false;
    }
#else
    if (requires_copy(buffer) || !is_f_contiguous(buffer)) {
        // If not F-contiguous or writeable or numpy's data let pybind handle the copy
        buffer = py::array_t<T, py::array::f_style | py::array::forcecast>::ensure(src);
        info = buffer.request();
        copy = false;
        strict = false;
    }
#endif

    return arma::Cube<T>(static_cast<T*>(info.ptr), info.shape[0], info.shape[1], info.shape[2], copy, strict);
} /* arr_to_mat */

/* The below functor approach is ported from:
 *     Arma_Wrapper - Paul Sangrey 2019
 *     Apache 2.0 License
 * This is a templated functor that has overloads that convert the various
 * types that I want to pass from Python to C++.
 */
template <typename returnT, typename SFINAE = std::true_type>
struct _to_arma {
    static_assert(!SFINAE::value, "The general case is not defined.");
    template <typename innerT>
    static returnT from(innerT&&);
}; /* to_arma */

template <typename returnT>
struct _to_arma<returnT, typename is_row<returnT>::type> {
    /* Overload concept on return type; convert to row */
    template <typename T>
    static returnT from(py::array_t<T>& arr, bool copy, bool strict) {
        return arr_to_row<T>(arr, copy, strict);
    }
}; /* to_arma */

template <typename returnT>
struct _to_arma<returnT, typename is_col<returnT>::type> {
    /* Overload concept on return type; convert to col */
    template <typename T>
    static returnT from(py::array_t<T>& arr, bool copy, bool strict) {
        return arr_to_col<T>(arr, copy, strict);
    }
}; /* to_arma */

template <typename returnT>
struct _to_arma<returnT, typename is_mat<returnT>::type> {
    /* Overload concept on return type; convert to matrix */
    template <typename T>
    static returnT from(py::array_t<T>& arr, bool copy, bool strict) {
        return arr_to_mat<T>(arr, copy, strict);
    }
}; /* to_arma */

template <typename returnT>
struct _to_arma<returnT, typename is_cube<returnT>::type> {
    /* Overload concept on return type; convert to cube */
    template <typename T>
    static returnT from(py::array_t<T>& arr, bool copy, bool strict) {
        return arr_to_cube<T>(arr, copy, strict);
    }
}; /* to_arma */

/*****************************************************************************************
 *                                   Armadillo to Numpy                                   *
 *****************************************************************************************/
/* ---------------------------------- array_constr ---------------------------------- */
template <typename T>
inline py::array_t<T> _construct_array(arma::Row<T>* data) {
    constexpr ssize_t tsize = static_cast<ssize_t>(sizeof(T));
    ssize_t ncols = static_cast<ssize_t>(data->n_cols);

    py::capsule base = create_capsule<arma::Row<T>>(data);

    return py::array_t<T>(
        {static_cast<ssize_t>(1), ncols},  // shape
        {tsize, tsize},                    // F-style contiguous strides
        data->memptr(),                    // the data pointer
        base                               // numpy array references this parent
    );
} /* _construct_array */

template <typename T>
inline py::array_t<T> _construct_array(arma::Col<T>* data) {
    constexpr ssize_t tsize = static_cast<ssize_t>(sizeof(T));
    ssize_t nrows = static_cast<ssize_t>(data->n_rows);

    py::capsule base = create_capsule<arma::Col<T>>(data);

    return py::array_t<T>(
        {nrows, static_cast<ssize_t>(1)},  // shape
        {tsize, nrows * tsize},            // F-style contiguous strides
        data->memptr(),                    // the data pointer
        base                               // numpy array references this parent
    );
} /* _construct_array */

template <typename T>
inline py::array_t<T> _construct_array(arma::Mat<T>* data) {
    constexpr ssize_t tsize = static_cast<ssize_t>(sizeof(T));
    ssize_t nrows = static_cast<ssize_t>(data->n_rows);
    ssize_t ncols = static_cast<ssize_t>(data->n_cols);

    py::capsule base = create_capsule<arma::Mat<T>>(data);

    return py::array_t<T>(
        {nrows, ncols},          // shape
        {tsize, nrows * tsize},  // F-style contiguous strides
        data->memptr(),          // the data pointer
        base                     // numpy array references this parent
    );
} /* _construct_array */

template <typename T>
inline py::array_t<T> _construct_array(arma::Cube<T>* data) {
    constexpr ssize_t tsize = static_cast<ssize_t>(sizeof(T));
    ssize_t nrows = static_cast<ssize_t>(data->n_rows);
    ssize_t ncols = static_cast<ssize_t>(data->n_cols);
    ssize_t nslices = static_cast<ssize_t>(data->n_slices);

    py::capsule base = create_capsule<arma::Cube<T>>(data);

    return py::array_t<T>(
        {nslices, nrows, ncols},                        // shape
        {tsize * nrows * ncols, tsize, nrows * tsize},  // F-style contiguous strides
        data->memptr(),                                 // the data pointer
        base                                            // numpy array references this parent
    );
} /* _construct_array */

/* -------------------------------- Type specific funcs -------------------------------- */
/* ######################################## Row ######################################## */
template <typename T>
inline py::array_t<T> row_to_arr(const arma::Row<T>& src) {
    /* Convert armadillo row to numpy array */
    arma::Row<T>* data = new arma::Row<T>(src);
    return _construct_array<T>(data);
} /* row_to_arr */

template <typename T>
inline py::array_t<T> row_to_arr(arma::Row<T>&& src) {
    /* Convert armadillo row to numpy array */
    arma::Row<T>* data = new arma::Row<T>(std::move(src));
    return _construct_array<T>(data);
} /* row_to_arr */

template <typename T>
inline py::array_t<T> row_to_arr(arma::Row<T>& src, bool copy = false) {
    /* Convert armadillo row to numpy array */
    arma::Row<T>* data;
    if (!copy) {
        data = new arma::Row<T>(std::move(src));
    } else {
        data = new arma::Row<T>(src.memptr(), src.n_elem, true);
    }
    return _construct_array<T>(data);
} /* row_to_arr */

template <typename T>
inline py::array_t<T> row_to_arr(arma::Row<T>* src, bool copy = false) {
    /* Convert armadillo row to numpy array */
    arma::Row<T>* data;
    if (!copy) {
        data = new arma::Row<T>(std::move(*src));
    } else {
        data = new arma::Row<T>(src->memptr(), src->n_elem, true);
    }
    return _construct_array<T>(data);
} /* row_to_arr */

template <typename T>
inline void update_array(arma::Row<T>& src, py::array_t<T>& arr) {
    /* Update underlying numpy array */
    arr.resize({static_cast<ssize_t>(1), static_cast<ssize_t>(src.n_elem)}, false);
} /* update_array */

template <typename T>
inline void update_array(arma::Row<T>&& src, py::array_t<T>& arr) {
    /* Update underlying numpy array */
    arr.resize({static_cast<ssize_t>(1), static_cast<ssize_t>(src.n_elem)}, false);
} /* update_array */

template <typename T>
inline void update_array(arma::Row<T>* src, py::array_t<T>& arr) {
    /* Update underlying numpy array */
    arr.resize({static_cast<ssize_t>(1), static_cast<ssize_t>(src->n_elem)}, false);
} /* update_array */

/* ######################################## Col ######################################## */
template <typename T>
inline py::array_t<T> col_to_arr(const arma::Col<T>& src) {
    /* Convert armadillo col to numpy array */
    arma::Col<T>* data = new arma::Col<T>(src);
    return _construct_array<T>(data);
} /* col_to_arr */

template <typename T>
inline py::array_t<T> col_to_arr(arma::Col<T>&& src) {
    /* Convert armadillo col to numpy array */
    arma::Col<T>* data = new arma::Col<T>(std::move(src));
    return _construct_array<T>(data);
} /* col_to_arr */

template <typename T>
inline py::array_t<T> col_to_arr(arma::Col<T>& src, bool copy = false) {
    /* Convert armadillo col to numpy array */
    arma::Col<T>* data;
    if (!copy) {
        data = new arma::Col<T>(std::move(src));
    } else {
        data = new arma::Col<T>(src.memptr(), src.n_elem, true);
    }
    return _construct_array<T>(data);
} /* col_to_arr */

template <typename T>
inline py::array_t<T> col_to_arr(arma::Col<T>* src, bool copy = false) {
    /* Convert armadillo col to numpy array */
    arma::Col<T>* data;
    if (!copy) {
        data = new arma::Col<T>(std::move(*src));
    } else {
        data = new arma::Col<T>(src->memptr(), src->n_elem, true);
    }
    return _construct_array<T>(data);
} /* col_to_arr */

template <typename T>
inline void update_array(arma::Col<T>& src, py::array_t<T>& arr) {
    /* Update underlying numpy array */
    arr.resize({static_cast<ssize_t>(src.n_elem), static_cast<ssize_t>(1)}, false);
} /* update_array */

template <typename T>
inline void update_array(arma::Col<T>&& src, py::array_t<T>& arr) {
    /* Update underlying numpy array */
    arr.resize({static_cast<ssize_t>(src.n_elem), static_cast<ssize_t>(1)}, false);
} /* update_array */

template <typename T>
inline void update_array(arma::Col<T>* src, py::array_t<T>& arr) {
    /* Update underlying numpy array */
    arr.resize({static_cast<ssize_t>(src->n_elem), static_cast<ssize_t>(1)}, false);
} /* update_array */

/* ######################################## Mat ######################################## */
template <typename T>
inline py::array_t<T> mat_to_arr(const arma::Mat<T>& src) {
    arma::Mat<T>* data = new arma::Mat<T>(src);
    return _construct_array<T>(data);
} /* mat_to_arr */

template <typename T>
inline py::array_t<T> mat_to_arr(arma::Mat<T>&& src) {
    arma::Mat<T>* data = new arma::Mat<T>(std::move(src));
    return _construct_array<T>(data);
} /* mat_to_arr */

template <typename T>
inline py::array_t<T> mat_to_arr(arma::Mat<T>& src, bool copy = false) {
    arma::Mat<T>* data;
    if (!copy) {
        data = new arma::Mat<T>(std::move(src));
    } else {
        data = new arma::Mat<T>(src.memptr(), src.n_rows, src.n_cols, true);
    }
    return _construct_array<T>(data);
} /* mat_to_arr */

template <typename T>
inline py::array_t<T> mat_to_arr(arma::Mat<T>* src, bool copy = false) {
    arma::Mat<T>* data;
    if (!copy) {
        data = new arma::Mat<T>(std::move(*src));
    } else {
        data = new arma::Mat<T>(src->memptr(), src->n_rows, src->n_cols, true);
    }
    return _construct_array<T>(data);
} /* mat_to_arr */

template <typename T>
inline void update_array(arma::Mat<T>&& src, py::array_t<T>& arr) {
    /* Update underlying numpy array */
    arr.resize({static_cast<ssize_t>(src.n_rows), static_cast<ssize_t>(src.n_cols)}, false);
} /* update_array */

template <typename T>
inline void update_array(arma::Mat<T>& src, py::array_t<T>& arr) {
    /* Update underlying numpy array */
    arr.resize({static_cast<ssize_t>(src.n_rows), static_cast<ssize_t>(src.n_cols)}, false);
} /* update_array */

template <typename T>
inline void update_array(arma::Mat<T>* src, py::array_t<T>& arr) {
    /* Update underlying numpy array */
    arr.resize({static_cast<ssize_t>(src->n_rows), static_cast<ssize_t>(src->n_cols)}, false);
} /* update_array */

/* ######################################## Cube ######################################## */
template <typename T>
inline py::array_t<T> cube_to_arr(const arma::Cube<T>& src) {
    arma::Cube<T>* data = new arma::Cube<T>(src);
    return _construct_array<T>(data);
} /* cube_to_arr */

template <typename T>
inline py::array_t<T> cube_to_arr(arma::Cube<T>&& src) {
    arma::Cube<T>* data = new arma::Cube<T>(std::move(src));
    return _construct_array<T>(data);
} /* cube_to_arr */

template <typename T>
inline py::array_t<T> cube_to_arr(arma::Cube<T>& src, bool copy = false) {
    arma::Cube<T>* data;
    if (!copy) {
        data = new arma::Cube<T>(std::move(src));
    } else {
        data = new arma::Cube<T>(src.memptr(), src.n_rows, src.n_cols, src.n_slices, true);
    }
    return _construct_array<T>(data);
} /* cube_to_arr */

template <typename T>
inline py::array_t<T> cube_to_arr(arma::Cube<T>* src, bool copy = false) {
    arma::Cube<T>* data;
    if (!copy) {
        data = new arma::Cube<T>(std::move(*src));
    } else {
        data = new arma::Cube<T>(src->memptr(), src->n_rows, src->n_cols, src->n_slices, true);
    }
    return _construct_array<T>(data);
} /* cube_to_arr */

template <typename T>
inline void update_array(arma::Cube<T>&& src, py::array_t<T>& arr) {
    arr.resize(
        {static_cast<ssize_t>(src.n_rows), static_cast<ssize_t>(src.n_cols), static_cast<ssize_t>(src.n_slices)},
        false);
} /* update_array */

template <typename T>
inline void update_array(arma::Cube<T>& src, py::array_t<T>& arr) {
    arr.resize(
        {static_cast<ssize_t>(src.n_rows), static_cast<ssize_t>(src.n_cols), static_cast<ssize_t>(src.n_slices)},
        false);
} /* update_array */

template <typename T>
inline void update_array(arma::Cube<T>* src, py::array_t<T>& arr) {
    arr.resize(
        {static_cast<ssize_t>(src->n_rows), static_cast<ssize_t>(src->n_cols), static_cast<ssize_t>(src->n_slices)},
        false);
} /* update_array */

/* ---------------------------------- to_numpy ---------------------------------- */
template <typename armaT, typename T = typename armaT::elem_type, is_Cube<armaT> = 0>
inline py::array_t<T> to_numpy(const armaT& src) {
    arma::Cube<T>* data = new arma::Cube<T>(src);
    return _construct_array<T>(data);
} /* cube_to_arr */

template <typename armaT, typename T = typename armaT::elem_type, is_Cube<armaT> = 0>
inline py::array_t<T> to_numpy(armaT&& src) {
    arma::Cube<T>* data = new arma::Cube<T>(std::forward<arma::Cube<T>>(src));
    return _construct_array<T>(data);
} /* cube_to_arr */

template <typename armaT, typename T = typename armaT::elem_type, is_Cube<armaT> = 0>
inline py::array_t<T> to_numpy(armaT& src, bool copy = false) {
    arma::Cube<T>* data;
    if (!copy) {
        data = new arma::Cube<T>(std::move(src));
    } else {
        data = new arma::Cube<T>(src.memptr(), src.n_rows, src.n_cols, src.n_slices, true);
    }
    return _construct_array<T>(data);
} /* cube_to_arr */

template <typename armaT, typename T = typename armaT::elem_type, is_Cube<armaT> = 0>
inline py::array_t<T> to_numpy(armaT* src, bool copy = false) {
    arma::Cube<T>* data;
    if (!copy) {
        data = new arma::Cube<T>(std::move(*src));
    } else {
        data = new arma::Cube<T>(src->memptr(), src->n_rows, src->n_cols, src->n_slices, true);
    }
    return _construct_array<T>(data);
} /* cube_to_arr */

template <typename armaT, typename T = typename armaT::elem_type, is_Mat<armaT> = 1>
inline py::array_t<T> to_numpy(const armaT& src) {
    // use armadillo copy constructor
    armaT* data = new armaT(src);
    return _construct_array<T>(data);
} /* to_numpy */

template <typename armaT, typename T = typename armaT::elem_type, is_Mat<armaT> = 1>
inline py::array_t<T> to_numpy(armaT&& src) {
    // steal mem
    armaT* data = new armaT(std::forward<armaT>(src));
    return _construct_array<T>(data);
} /* to_numpy */

template <typename armaT, typename T = typename armaT::elem_type, is_Mat_only<armaT> = 2>
inline py::array_t<T> to_numpy(armaT& src, bool copy = false) {
    // if not copy we steal
    armaT* data;
    if (!copy) {
        data = new armaT(std::move(src));
    } else {
        data = new armaT(src.memptr(), src.n_rows, src.n_cols, true);
    }
    return _construct_array<T>(data);
} /* to_numpy */

template <typename armaT, typename T = typename armaT::elem_type, is_Mat_only<armaT> = 2>
inline py::array_t<T> to_numpy(armaT* src, bool copy = false) {
    // if not copy we steal
    armaT* data;
    if (!copy) {
        data = new armaT(std::move(*src));
    } else {
        data = new armaT(src->memptr(), src->n_rows, src->n_cols, true);
    }
    return _construct_array<T>(data);
} /* to_numpy */

template <typename armaT, typename T = typename armaT::elem_type, is_Vec<armaT> = 3>
inline py::array_t<T> to_numpy(armaT& src, bool copy = false) {
    // if not copy we steal
    armaT* data;
    if (!copy) {
        data = new armaT(std::move(src));
    } else {
        data = new armaT(src.memptr(), src.n_elem, true);
    }
    return _construct_array<T>(data);
} /* to_numpy */

template <typename armaT, typename T = typename armaT::elem_type, is_Vec<armaT> = 3>
inline py::array_t<T> to_numpy(armaT* src, bool copy = false) {
    // if not copy we steal
    armaT* data;
    if (!copy) {
        data = new armaT(std::move(*src));
    } else {
        data = new armaT(src->memptr(), src->n_elem, true);
    }
    return _construct_array<T>(data);
} /* to_numpy */

}  // namespace carma

NAMESPACE_BEGIN(pybind11)
NAMESPACE_BEGIN(detail)

template <typename armaT>
struct type_caster<armaT, enable_if_t<carma::is_convertible<armaT>::value>> {
    using T = typename armaT::elem_type;

    /* Convert numpy array to Armadillo Matrix
     *
     * The default behaviour is to avoid copying, we copy if:
     * - ndim == 2 && not F contiguous memory
     * - writeable is false
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
        if (info.ptr == nullptr) {
            return false;
        }

        value = carma::_to_arma<armaT>::from(buffer, copy, strict);
        return true;
    }

   private:
    // Cast implementation
    static handle cast_impl(armaT&& src, return_value_policy policy, handle) {
        switch (policy) {
            case return_value_policy::move:
                return carma::to_numpy<armaT>(std::move(src)).release();
            case return_value_policy::automatic:
                return carma::to_numpy<armaT>(std::move(src)).release();
            case return_value_policy::take_ownership:
                return carma::to_numpy<armaT>(std::move(src)).release();
            case return_value_policy::copy:
                return carma::to_numpy<armaT>(src, true).release();
            default:
                throw cast_error("unhandled return_value_policy");
        }
    }

    static handle cast_impl(armaT* src, return_value_policy policy, handle) {
        switch (policy) {
            case return_value_policy::move:
                return carma::to_numpy<armaT>(src).release();
            case return_value_policy::automatic:
                return carma::to_numpy<armaT>(src).release();
            case return_value_policy::take_ownership:
                return carma::to_numpy<armaT>(src).release();
            case return_value_policy::copy:
                return carma::to_numpy<armaT>(src, true).release();
            default:
                throw cast_error("unhandled return_value_policy");
        }
    }

   public:
    // Normal returned non-reference, non-const value: we steal
    static handle cast(armaT&& src, return_value_policy policy, handle parent) {
        return cast_impl(std::move(src), policy, parent);
    }
    // If you return a non-reference const; we copy
    static handle cast(const armaT&& src, return_value_policy policy, handle parent) {
        policy = return_value_policy::copy;
        return cast_impl(&src, policy, parent);
    }
    // lvalue reference return; default (automatic) becomes steal
    static handle cast(armaT& src, return_value_policy policy, handle parent) {
        return cast_impl(&src, policy, parent);
    }
    // const lvalue reference return; default (automatic) becomes copy
    static handle cast(const armaT& src, return_value_policy policy, handle parent) {
        policy = return_value_policy::copy;
        return cast_impl(&src, policy, parent);
    }
    // non-const pointer return; we steal
    static handle cast(armaT* src, return_value_policy policy, handle parent) { return cast_impl(src, policy, parent); }
    // const pointer return; we copy
    static handle cast(const armaT* src, return_value_policy policy, handle parent) {
        policy = return_value_policy::copy;
        return cast_impl(src, policy, parent);
    }

    PYBIND11_TYPE_CASTER(armaT, _("Numpy.ndarray[") + npy_format_descriptor<T>::name + _("]"));
};
NAMESPACE_END(detail)
NAMESPACE_END(pybind11)
#endif /* ARMA_CONVERTERS */
