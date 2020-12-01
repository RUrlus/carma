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

#define IS_CONVERTIBLE_FIXED(type) \
    template <>                    \
    struct is_convertible<type> : std::true_type {}

#define IS_MAT_FIXED(type) \
    template <>            \
    struct is_mat<type> : std::true_type {}

#define IS_COL_FIXED(type) \
    template <>            \
    struct is_col<type> : std::true_type {}

#define IS_ROW_FIXED(type) \
    template <>            \
    struct is_row<type> : std::true_type {}

namespace carma {

// Base template:
template <typename T>
struct is_convertible : std::false_type {};

// Specialisations:
template <typename T>
struct is_convertible<arma::Mat<T>> : std::true_type {};
template <typename T>
struct is_convertible<arma::Col<T>> : std::true_type {};
template <typename T>
struct is_convertible<arma::Row<T>> : std::true_type {};
template <typename T>
struct is_convertible<arma::Cube<T>> : std::true_type {};

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

/* convertible */
IS_CONVERTIBLE_FIXED(arma::umat22);
IS_CONVERTIBLE_FIXED(arma::umat33);
IS_CONVERTIBLE_FIXED(arma::umat44);
IS_CONVERTIBLE_FIXED(arma::umat55);
IS_CONVERTIBLE_FIXED(arma::umat66);
IS_CONVERTIBLE_FIXED(arma::umat77);
IS_CONVERTIBLE_FIXED(arma::umat88);
IS_CONVERTIBLE_FIXED(arma::umat99);
IS_CONVERTIBLE_FIXED(arma::imat22);
IS_CONVERTIBLE_FIXED(arma::imat33);
IS_CONVERTIBLE_FIXED(arma::imat44);
IS_CONVERTIBLE_FIXED(arma::imat55);
IS_CONVERTIBLE_FIXED(arma::imat66);
IS_CONVERTIBLE_FIXED(arma::imat77);
IS_CONVERTIBLE_FIXED(arma::imat88);
IS_CONVERTIBLE_FIXED(arma::imat99);
IS_CONVERTIBLE_FIXED(arma::fmat22);
IS_CONVERTIBLE_FIXED(arma::fmat33);
IS_CONVERTIBLE_FIXED(arma::fmat44);
IS_CONVERTIBLE_FIXED(arma::fmat55);
IS_CONVERTIBLE_FIXED(arma::fmat66);
IS_CONVERTIBLE_FIXED(arma::fmat77);
IS_CONVERTIBLE_FIXED(arma::fmat88);
IS_CONVERTIBLE_FIXED(arma::fmat99);
IS_CONVERTIBLE_FIXED(arma::mat22);
IS_CONVERTIBLE_FIXED(arma::mat33);
IS_CONVERTIBLE_FIXED(arma::mat44);
IS_CONVERTIBLE_FIXED(arma::mat55);
IS_CONVERTIBLE_FIXED(arma::mat66);
IS_CONVERTIBLE_FIXED(arma::mat77);
IS_CONVERTIBLE_FIXED(arma::mat88);
IS_CONVERTIBLE_FIXED(arma::mat99);
IS_CONVERTIBLE_FIXED(arma::cx_fmat22);
IS_CONVERTIBLE_FIXED(arma::cx_fmat33);
IS_CONVERTIBLE_FIXED(arma::cx_fmat44);
IS_CONVERTIBLE_FIXED(arma::cx_fmat55);
IS_CONVERTIBLE_FIXED(arma::cx_fmat66);
IS_CONVERTIBLE_FIXED(arma::cx_fmat77);
IS_CONVERTIBLE_FIXED(arma::cx_fmat88);
IS_CONVERTIBLE_FIXED(arma::cx_fmat99);
IS_CONVERTIBLE_FIXED(arma::cx_mat22);
IS_CONVERTIBLE_FIXED(arma::cx_mat33);
IS_CONVERTIBLE_FIXED(arma::cx_mat44);
IS_CONVERTIBLE_FIXED(arma::cx_mat55);
IS_CONVERTIBLE_FIXED(arma::cx_mat66);
IS_CONVERTIBLE_FIXED(arma::cx_mat77);
IS_CONVERTIBLE_FIXED(arma::cx_mat88);
IS_CONVERTIBLE_FIXED(arma::cx_mat99);
IS_CONVERTIBLE_FIXED(arma::uvec2);
IS_CONVERTIBLE_FIXED(arma::uvec3);
IS_CONVERTIBLE_FIXED(arma::uvec4);
IS_CONVERTIBLE_FIXED(arma::uvec5);
IS_CONVERTIBLE_FIXED(arma::uvec6);
IS_CONVERTIBLE_FIXED(arma::uvec7);
IS_CONVERTIBLE_FIXED(arma::uvec8);
IS_CONVERTIBLE_FIXED(arma::uvec9);
IS_CONVERTIBLE_FIXED(arma::ivec2);
IS_CONVERTIBLE_FIXED(arma::ivec3);
IS_CONVERTIBLE_FIXED(arma::ivec4);
IS_CONVERTIBLE_FIXED(arma::ivec5);
IS_CONVERTIBLE_FIXED(arma::ivec6);
IS_CONVERTIBLE_FIXED(arma::ivec7);
IS_CONVERTIBLE_FIXED(arma::ivec8);
IS_CONVERTIBLE_FIXED(arma::ivec9);
IS_CONVERTIBLE_FIXED(arma::fvec2);
IS_CONVERTIBLE_FIXED(arma::fvec3);
IS_CONVERTIBLE_FIXED(arma::fvec4);
IS_CONVERTIBLE_FIXED(arma::fvec5);
IS_CONVERTIBLE_FIXED(arma::fvec6);
IS_CONVERTIBLE_FIXED(arma::fvec7);
IS_CONVERTIBLE_FIXED(arma::fvec8);
IS_CONVERTIBLE_FIXED(arma::fvec9);
IS_CONVERTIBLE_FIXED(arma::vec2);
IS_CONVERTIBLE_FIXED(arma::vec3);
IS_CONVERTIBLE_FIXED(arma::vec4);
IS_CONVERTIBLE_FIXED(arma::vec5);
IS_CONVERTIBLE_FIXED(arma::vec6);
IS_CONVERTIBLE_FIXED(arma::vec7);
IS_CONVERTIBLE_FIXED(arma::vec8);
IS_CONVERTIBLE_FIXED(arma::vec9);
IS_CONVERTIBLE_FIXED(arma::cx_fvec2);
IS_CONVERTIBLE_FIXED(arma::cx_fvec3);
IS_CONVERTIBLE_FIXED(arma::cx_fvec4);
IS_CONVERTIBLE_FIXED(arma::cx_fvec5);
IS_CONVERTIBLE_FIXED(arma::cx_fvec6);
IS_CONVERTIBLE_FIXED(arma::cx_fvec7);
IS_CONVERTIBLE_FIXED(arma::cx_fvec8);
IS_CONVERTIBLE_FIXED(arma::cx_fvec9);
IS_CONVERTIBLE_FIXED(arma::cx_vec2);
IS_CONVERTIBLE_FIXED(arma::cx_vec3);
IS_CONVERTIBLE_FIXED(arma::cx_vec4);
IS_CONVERTIBLE_FIXED(arma::cx_vec5);
IS_CONVERTIBLE_FIXED(arma::cx_vec6);
IS_CONVERTIBLE_FIXED(arma::cx_vec7);
IS_CONVERTIBLE_FIXED(arma::cx_vec8);
IS_CONVERTIBLE_FIXED(arma::cx_vec9);
IS_CONVERTIBLE_FIXED(arma::urowvec2);
IS_CONVERTIBLE_FIXED(arma::urowvec3);
IS_CONVERTIBLE_FIXED(arma::urowvec4);
IS_CONVERTIBLE_FIXED(arma::urowvec5);
IS_CONVERTIBLE_FIXED(arma::urowvec6);
IS_CONVERTIBLE_FIXED(arma::urowvec7);
IS_CONVERTIBLE_FIXED(arma::urowvec8);
IS_CONVERTIBLE_FIXED(arma::urowvec9);
IS_CONVERTIBLE_FIXED(arma::irowvec2);
IS_CONVERTIBLE_FIXED(arma::irowvec3);
IS_CONVERTIBLE_FIXED(arma::irowvec4);
IS_CONVERTIBLE_FIXED(arma::irowvec5);
IS_CONVERTIBLE_FIXED(arma::irowvec6);
IS_CONVERTIBLE_FIXED(arma::irowvec7);
IS_CONVERTIBLE_FIXED(arma::irowvec8);
IS_CONVERTIBLE_FIXED(arma::irowvec9);
IS_CONVERTIBLE_FIXED(arma::frowvec2);
IS_CONVERTIBLE_FIXED(arma::frowvec3);
IS_CONVERTIBLE_FIXED(arma::frowvec4);
IS_CONVERTIBLE_FIXED(arma::frowvec5);
IS_CONVERTIBLE_FIXED(arma::frowvec6);
IS_CONVERTIBLE_FIXED(arma::frowvec7);
IS_CONVERTIBLE_FIXED(arma::frowvec8);
IS_CONVERTIBLE_FIXED(arma::frowvec9);
IS_CONVERTIBLE_FIXED(arma::rowvec2);
IS_CONVERTIBLE_FIXED(arma::rowvec3);
IS_CONVERTIBLE_FIXED(arma::rowvec4);
IS_CONVERTIBLE_FIXED(arma::rowvec5);
IS_CONVERTIBLE_FIXED(arma::rowvec6);
IS_CONVERTIBLE_FIXED(arma::rowvec7);
IS_CONVERTIBLE_FIXED(arma::rowvec8);
IS_CONVERTIBLE_FIXED(arma::rowvec9);
IS_CONVERTIBLE_FIXED(arma::cx_frowvec2);
IS_CONVERTIBLE_FIXED(arma::cx_frowvec3);
IS_CONVERTIBLE_FIXED(arma::cx_frowvec4);
IS_CONVERTIBLE_FIXED(arma::cx_frowvec5);
IS_CONVERTIBLE_FIXED(arma::cx_frowvec6);
IS_CONVERTIBLE_FIXED(arma::cx_frowvec7);
IS_CONVERTIBLE_FIXED(arma::cx_frowvec8);
IS_CONVERTIBLE_FIXED(arma::cx_frowvec9);
IS_CONVERTIBLE_FIXED(arma::cx_rowvec2);
IS_CONVERTIBLE_FIXED(arma::cx_rowvec3);
IS_CONVERTIBLE_FIXED(arma::cx_rowvec4);
IS_CONVERTIBLE_FIXED(arma::cx_rowvec5);
IS_CONVERTIBLE_FIXED(arma::cx_rowvec6);
IS_CONVERTIBLE_FIXED(arma::cx_rowvec7);
IS_CONVERTIBLE_FIXED(arma::cx_rowvec8);
IS_CONVERTIBLE_FIXED(arma::cx_rowvec9);

/* mat */
IS_MAT_FIXED(arma::umat22);
IS_MAT_FIXED(arma::umat33);
IS_MAT_FIXED(arma::umat44);
IS_MAT_FIXED(arma::umat55);
IS_MAT_FIXED(arma::umat66);
IS_MAT_FIXED(arma::umat77);
IS_MAT_FIXED(arma::umat88);
IS_MAT_FIXED(arma::umat99);
IS_MAT_FIXED(arma::imat22);
IS_MAT_FIXED(arma::imat33);
IS_MAT_FIXED(arma::imat44);
IS_MAT_FIXED(arma::imat55);
IS_MAT_FIXED(arma::imat66);
IS_MAT_FIXED(arma::imat77);
IS_MAT_FIXED(arma::imat88);
IS_MAT_FIXED(arma::imat99);
IS_MAT_FIXED(arma::fmat22);
IS_MAT_FIXED(arma::fmat33);
IS_MAT_FIXED(arma::fmat44);
IS_MAT_FIXED(arma::fmat55);
IS_MAT_FIXED(arma::fmat66);
IS_MAT_FIXED(arma::fmat77);
IS_MAT_FIXED(arma::fmat88);
IS_MAT_FIXED(arma::fmat99);
IS_MAT_FIXED(arma::mat22);
IS_MAT_FIXED(arma::mat33);
IS_MAT_FIXED(arma::mat44);
IS_MAT_FIXED(arma::mat55);
IS_MAT_FIXED(arma::mat66);
IS_MAT_FIXED(arma::mat77);
IS_MAT_FIXED(arma::mat88);
IS_MAT_FIXED(arma::mat99);
IS_MAT_FIXED(arma::cx_fmat22);
IS_MAT_FIXED(arma::cx_fmat33);
IS_MAT_FIXED(arma::cx_fmat44);
IS_MAT_FIXED(arma::cx_fmat55);
IS_MAT_FIXED(arma::cx_fmat66);
IS_MAT_FIXED(arma::cx_fmat77);
IS_MAT_FIXED(arma::cx_fmat88);
IS_MAT_FIXED(arma::cx_fmat99);
IS_MAT_FIXED(arma::cx_mat22);
IS_MAT_FIXED(arma::cx_mat33);
IS_MAT_FIXED(arma::cx_mat44);
IS_MAT_FIXED(arma::cx_mat55);
IS_MAT_FIXED(arma::cx_mat66);
IS_MAT_FIXED(arma::cx_mat77);
IS_MAT_FIXED(arma::cx_mat88);
IS_MAT_FIXED(arma::cx_mat99);

/* vec */
IS_COL_FIXED(arma::uvec2);
IS_COL_FIXED(arma::uvec3);
IS_COL_FIXED(arma::uvec4);
IS_COL_FIXED(arma::uvec5);
IS_COL_FIXED(arma::uvec6);
IS_COL_FIXED(arma::uvec7);
IS_COL_FIXED(arma::uvec8);
IS_COL_FIXED(arma::uvec9);
IS_COL_FIXED(arma::ivec2);
IS_COL_FIXED(arma::ivec3);
IS_COL_FIXED(arma::ivec4);
IS_COL_FIXED(arma::ivec5);
IS_COL_FIXED(arma::ivec6);
IS_COL_FIXED(arma::ivec7);
IS_COL_FIXED(arma::ivec8);
IS_COL_FIXED(arma::ivec9);
IS_COL_FIXED(arma::fvec2);
IS_COL_FIXED(arma::fvec3);
IS_COL_FIXED(arma::fvec4);
IS_COL_FIXED(arma::fvec5);
IS_COL_FIXED(arma::fvec6);
IS_COL_FIXED(arma::fvec7);
IS_COL_FIXED(arma::fvec8);
IS_COL_FIXED(arma::fvec9);
IS_COL_FIXED(arma::vec2);
IS_COL_FIXED(arma::vec3);
IS_COL_FIXED(arma::vec4);
IS_COL_FIXED(arma::vec5);
IS_COL_FIXED(arma::vec6);
IS_COL_FIXED(arma::vec7);
IS_COL_FIXED(arma::vec8);
IS_COL_FIXED(arma::vec9);
IS_COL_FIXED(arma::cx_fvec2);
IS_COL_FIXED(arma::cx_fvec3);
IS_COL_FIXED(arma::cx_fvec4);
IS_COL_FIXED(arma::cx_fvec5);
IS_COL_FIXED(arma::cx_fvec6);
IS_COL_FIXED(arma::cx_fvec7);
IS_COL_FIXED(arma::cx_fvec8);
IS_COL_FIXED(arma::cx_fvec9);
IS_COL_FIXED(arma::cx_vec2);
IS_COL_FIXED(arma::cx_vec3);
IS_COL_FIXED(arma::cx_vec4);
IS_COL_FIXED(arma::cx_vec5);
IS_COL_FIXED(arma::cx_vec6);
IS_COL_FIXED(arma::cx_vec7);
IS_COL_FIXED(arma::cx_vec8);
IS_COL_FIXED(arma::cx_vec9);

/* row */
IS_ROW_FIXED(arma::urowvec2);
IS_ROW_FIXED(arma::urowvec3);
IS_ROW_FIXED(arma::urowvec4);
IS_ROW_FIXED(arma::urowvec5);
IS_ROW_FIXED(arma::urowvec6);
IS_ROW_FIXED(arma::urowvec7);
IS_ROW_FIXED(arma::urowvec8);
IS_ROW_FIXED(arma::urowvec9);
IS_ROW_FIXED(arma::irowvec2);
IS_ROW_FIXED(arma::irowvec3);
IS_ROW_FIXED(arma::irowvec4);
IS_ROW_FIXED(arma::irowvec5);
IS_ROW_FIXED(arma::irowvec6);
IS_ROW_FIXED(arma::irowvec7);
IS_ROW_FIXED(arma::irowvec8);
IS_ROW_FIXED(arma::irowvec9);
IS_ROW_FIXED(arma::frowvec2);
IS_ROW_FIXED(arma::frowvec3);
IS_ROW_FIXED(arma::frowvec4);
IS_ROW_FIXED(arma::frowvec5);
IS_ROW_FIXED(arma::frowvec6);
IS_ROW_FIXED(arma::frowvec7);
IS_ROW_FIXED(arma::frowvec8);
IS_ROW_FIXED(arma::frowvec9);
IS_ROW_FIXED(arma::rowvec2);
IS_ROW_FIXED(arma::rowvec3);
IS_ROW_FIXED(arma::rowvec4);
IS_ROW_FIXED(arma::rowvec5);
IS_ROW_FIXED(arma::rowvec6);
IS_ROW_FIXED(arma::rowvec7);
IS_ROW_FIXED(arma::rowvec8);
IS_ROW_FIXED(arma::rowvec9);
IS_ROW_FIXED(arma::cx_frowvec2);
IS_ROW_FIXED(arma::cx_frowvec3);
IS_ROW_FIXED(arma::cx_frowvec4);
IS_ROW_FIXED(arma::cx_frowvec5);
IS_ROW_FIXED(arma::cx_frowvec6);
IS_ROW_FIXED(arma::cx_frowvec7);
IS_ROW_FIXED(arma::cx_frowvec8);
IS_ROW_FIXED(arma::cx_frowvec9);
IS_ROW_FIXED(arma::cx_rowvec2);
IS_ROW_FIXED(arma::cx_rowvec3);
IS_ROW_FIXED(arma::cx_rowvec4);
IS_ROW_FIXED(arma::cx_rowvec5);
IS_ROW_FIXED(arma::cx_rowvec6);
IS_ROW_FIXED(arma::cx_rowvec7);
IS_ROW_FIXED(arma::cx_rowvec8);
IS_ROW_FIXED(arma::cx_rowvec9);

enum class Deallocator { Undefined, Arma, Free, Delete, None };

// Not a struct to force all fields initialization
template <typename armaT>
class Data {
   public:
    Data(armaT* data, Deallocator deallocator) : data(data), deallocator(deallocator) {}
    armaT* data;
    Deallocator deallocator;
};

template <typename armaT, typename = std::enable_if_t<is_convertible<armaT>::value>>
inline Data<typename std::decay_t<typename armaT::elem_type>> copy_mem(armaT& src) {
    using T = typename armaT::elem_type;
    size_t N = src.n_elem;
    T* data = new T[N];
    std::memcpy(data, src.memptr(), sizeof(T) * N);
    return {data, Deallocator::Delete};
}

template <typename armaT, typename = std::enable_if_t<is_convertible<armaT>::value>>
inline Data<typename std::decay_t<typename armaT::elem_type>> steal_mem(armaT* src) {
    using T = typename armaT::elem_type;
    T* data = src->memptr();
    arma::access::rw(src->mem) = 0;
    return {data, Deallocator::Arma};
}

template <typename armaT, typename = std::enable_if_t<is_convertible<armaT>::value>>
inline Data<typename armaT::elem_type> get_data(armaT* src, bool copy) {
    using T = typename armaT::elem_type;
    if (copy) {
        size_t N = src->n_elem;
        T* data = new T[N];
        std::memcpy(data, src->memptr(), sizeof(T) * N);
        return {data, Deallocator::Delete};
    } else {
        T* data = src->memptr();
        arma::access::rw(src->mem) = 0;
        return {data, Deallocator::Arma};
    }
} /* get_data */

template <typename T>
inline py::capsule create_capsule(Data<T>& data) {
    /* Create a Python object that will free the allocated
     * memory when destroyed:
     */
    switch (data.deallocator) {
        case Deallocator::Arma:
            return py::capsule(data.data, [](void* f) {
                T* data = reinterpret_cast<T*>(f);
#ifndef NDEBUG
                // if in debug mode let us know what pointer is being freed
                std::cerr << "freeing copy memory @ " << f << std::endl;
#endif
                arma::memory::release(data);
            });
        case Deallocator::Free:
            return py::capsule(data.data, [](void* f) {
                T* data = reinterpret_cast<T*>(f);
#ifndef NDEBUG
                // if in debug mode let us know what pointer is being freed
                std::cerr << "freeing copy memory @ " << f << std::endl;
#endif
                free(data);
            });
        case Deallocator::Delete:
            return py::capsule(data.data, [](void* f) {
                T* data = reinterpret_cast<T*>(f);
#ifndef NDEBUG
                // if in debug mode let us know what pointer is being freed
                std::cerr << "freeing copy memory @ " << f << std::endl;
#endif
                delete[] data;
            });
        case Deallocator::Undefined:
            assert(false);
        case Deallocator::None:
        default:
            return py::capsule(data.data, [](void* f) {
                T* data = reinterpret_cast<T*>(f);
#ifndef NDEBUG
                // if in debug mode let us know what pointer is being freed
                std::cerr << "freeing copy memory @ " << f << std::endl;
#endif
            });
    }
} /* create_capsule */

template <typename T>
inline py::capsule create_dummy_capsule(T* data) {
    /* Create a Python object that will free the allocated
     * memory when destroyed:
     */
    return py::capsule(data, [](void* f) {
#ifndef NDEBUG
        // if in debug mode let us know what pointer is being freed
        std::cerr << "freeing memory @ " << f << std::endl;
#endif
    });
} /* create_dummy_capsule */

}  // namespace carma
#endif /* ARMA_UTILS */
