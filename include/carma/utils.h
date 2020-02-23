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
#include <utility>
#include <type_traits>

/* External headers */
#include <armadillo>
#include <pybind11/pybind11.h>
#include <pybind11/numpy.h>
namespace py = pybind11;

#ifndef ARMA_UTILS
#define ARMA_UTILS

namespace carma {

    // Base template:
    template <typename T>
    struct is_convertible : std::false_type {};

    // Specialisations:
    template<typename T> struct is_convertible<arma::Mat<T>> : std::true_type {};
    template<typename T> struct is_convertible<arma::Col<T>> : std::true_type {};
    template<typename T> struct is_convertible<arma::Row<T>> : std::true_type {};
    template<typename T> struct is_convertible<arma::Cube<T>> : std::true_type {};

    template<typename T> struct is_mat : std::false_type {};
    template<typename T> struct is_mat<arma::Mat<T>> : std::true_type {};

    template<typename T> struct is_col : std::false_type {};
    template<typename T> struct is_col<arma::Col<T>> : std::true_type {};

    template<typename T> struct is_row : std::false_type {};
    template<typename T> struct is_row<arma::Row<T>> : std::true_type {};

    template<typename T> struct is_cube : std::false_type {};
    template<typename T> struct is_cube<arma::Cube<T>> : std::true_type {};

    template <typename armaT, typename = std::enable_if_t<is_convertible<armaT>::value>>
    inline typename std::decay_t<typename armaT::elem_type> * copy_mem (armaT & src) {
        using T = typename armaT::elem_type;
        size_t N = src.n_elem;
        T * data = new T[N];
        std::memcpy(data, src.memptr(), sizeof(T) * N);
        return data;
    }

    template <typename armaT, typename = std::enable_if_t<is_convertible<armaT>::value>>
    inline typename std::decay_t<typename armaT::elem_type> * steal_mem(armaT & src) {
        using T = typename armaT::elem_type;
        T * data = src->memptr();
        arma::access::rw(src->mem) = 0;
        return data;
    }

    template <typename armaT, typename = std::enable_if_t<is_convertible<armaT>::value>>
    inline typename armaT::elem_type * get_data(armaT * src, bool copy) {
        using T = typename armaT::elem_type;
        if (copy) {
            size_t N = src->n_elem;
            T * data = new T[N];
            std::memcpy(data, src->memptr(), sizeof(T) * N);
            return data;
        } else {
            T * data = src->memptr();
            arma::access::rw(src->mem) = 0;
            return data;
        }
    } /* get_data */

    template <typename T>
    inline py::capsule create_capsule(T * data) {
        /* Create a Python object that will free the allocated
         * memory when destroyed:
         */
        py::capsule base(data, [](void *f) {
            T *data = reinterpret_cast<T *>(f);
            #ifndef NDEBUG
            // if in debug mode let us know what pointer is being freed
            std::cerr << "freeing memory @ " << f << std::endl;
            #endif
            delete[] data;
        });
        return base;
    } /* create_capsule */

} /* carma */
#endif /* ARMA_UTILS */
