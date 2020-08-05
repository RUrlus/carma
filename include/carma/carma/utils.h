/*  carma/utils.h: Utility functions for arma converters
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
#include <pybind11/numpy.h>
#include <pybind11/pybind11.h>
#include <armadillo>
namespace py = pybind11;

#ifndef ARMA_UTILS
#define ARMA_UTILS

namespace carma {

struct conversion_error : public std::exception {
    const char* _message;
    conversion_error(const char* message) : _message(message) {}
    const char* what() const throw() { return _message; }
};

// Mat catches Row and Col as well
template <typename T>
struct is_convertible {
    static const bool value = (arma::is_Mat<T>::value || arma::is_Cube<T>::value);
};

template <typename T>
struct _is_Vec {
    static const bool value = (arma::is_Row<T>::value || arma::is_Col<T>::value);
};

// for reference see: https://www.fluentcpp.com/2019/08/23/how-to-make-sfinae-pretty-and-robust/
template <typename armaT>
using is_Cube = std::enable_if_t<arma::is_Cube<armaT>::value, int>;
template <typename armaT>
using is_Vec = std::enable_if_t<_is_Vec<armaT>::value, int>;
template <typename armaT>
using is_Mat = std::enable_if_t<arma::is_Mat<armaT>::value, int>;
template <typename armaT>
using is_Mat_only = std::enable_if_t<arma::is_Mat_only<armaT>::value, int>;

template <typename T>
struct is_mat : std::false_type {};
template <typename T>
struct is_mat<arma::Mat<T>> : std::true_type {};

template <typename T>
struct is_col : std::false_type {};
template <typename T>
struct is_col<arma::Col<T>> : std::true_type {};

template <typename T>
struct is_row : std::false_type {};
template <typename T>
struct is_row<arma::Row<T>> : std::true_type {};

template <typename T>
struct is_cube : std::false_type {};
template <typename T>
struct is_cube<arma::Cube<T>> : std::true_type {};

template <typename armaT>
inline py::capsule create_capsule(armaT* data) {
    return py::capsule(data, [](void* f) {
        armaT* mat = reinterpret_cast<armaT*>(f);
#ifndef NDEBUG
        // if in debug mode let us know what pointer is being freed
        std::cerr << "freeing memory @ " << mat->memptr() << std::endl;
#endif
        delete mat;
    });
} /* create_capsule */

template <typename armaT>
inline py::capsule create_dummy_capsule(armaT* data) {
    return py::capsule(data->memptr(), [](void* f) {
#ifndef NDEBUG
        // if in debug mode let us know what pointer is being freed
        std::cerr << "dummy capsule for memory @ " << f << std::endl;
#endif
    });
} /* create_capsule */

}  // namespace carma
#endif /* ARMA_UTILS */
