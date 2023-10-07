/*  carma_bits/numpy_alloc.hpp: Wrapper around PyDataMem_* for use with Armadillo
 *  Copyright (c) 2022 Ralph Urlus <rurlus.dev@gmail.com>
 *  All rights reserved. Use of this source code is governed by a
 *  Apache-2.0 license that can be found in the LICENSE file.
 */
#pragma once

// pybind11 include required even if not explicitly used
// to prevent link with pythonXX_d.lib on Win32
// (cf Py_DEBUG defined in numpy headers and https://github.com/pybind/pybind11/issues/1295)
#include <pybind11/pybind11.h>
#define NPY_NO_DEPRECATED_API NPY_1_18_API_VERSION
#include <numpy/arrayobject.h>
#include <numpy/ndarraytypes.h>

#include <carma_bits/numpy_api.hpp>
#include <cstddef>
#ifdef CARMA_DEBUG
#include <iostream>
#endif

namespace carma {
namespace alloc {

inline void* npy_malloc(size_t bytes) {
    const auto& api = internal::npy_api::get();
    void* ptr = api.PyDataMem_NEW_(bytes);
#ifdef CARMA_EXTRA_DEBUG
    std::cout << "|carma| allocated " << ptr << "\n";
#endif  // ARMA_EXTRA_DEBUG
    return ptr;
}  // npy_malloc

inline void npy_free(void* ptr) {
    const auto& api = internal::npy_api::get();
#ifdef CARMA_EXTRA_DEBUG
    std::cout << "|carma| freeing " << ptr << "\n";
#endif  // CARMA_EXTRA_DEBUG
    api.PyDataMem_FREE_(ptr);
}  // npy_free

}  // namespace alloc
}  // namespace carma

// carma makes use of the below Armadillo macros to enable
// handing over memory ownership to armadillo objects.
// These definitions must be set before armadillo is included.
#define ARMA_ALIEN_MEM_ALLOC_FUNCTION carma::alloc::npy_malloc
#define ARMA_ALIEN_MEM_FREE_FUNCTION carma::alloc::npy_free
#ifndef CARMA_ARMA_ALIEN_MEM_FUNCTIONS_SET
#define CARMA_ARMA_ALIEN_MEM_FUNCTIONS_SET true
#endif
