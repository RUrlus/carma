#ifndef CNALLOC_INCLUDE_CNALLOC_HPP_
#define CNALLOC_INCLUDE_CNALLOC_HPP_

#define NPY_NO_DEPRECATED_API NPY_1_14_API_VERSION
#include <numpy/arrayobject.h>
#include <numpy/ndarraytypes.h>

#include <cstddef>
#ifdef CARMA_DEV_DEBUG
#include <iostream>
#endif

namespace cnalloc {

inline void* npy_malloc(std::size_t bytes) {
    if (PyArray_API == NULL) {
        _import_array();
    }
#ifdef CARMA_DEV_DEBUG
    std::cout << "\n-----------\nCARMA DEBUG\n-----------\n";
    std::cout << "Using numpy allocator" << "\n";
    std::cout << "-----------\n";
#endif  // ARMA_EXTRA_DEBUG
    return PyDataMem_NEW(bytes);
} // npy_malloc

inline void npy_free(void* ptr) {
    if (PyArray_API == NULL) {
        _import_array();
    }
#ifdef CARMA_DEV_DEBUG
    std::cout << "\n-----------\nCARMA DEBUG\n-----------\n";
    std::cerr << "Using numpy deallocator\n";
    std::cout << "-----------\n";
#endif  // ARMA_EXTRA_DEBUG
    PyDataMem_FREE(ptr);
} // npy_free

} // namespace cnalloc

#define ARMA_ALIEN_MEM_ALLOC_FUNCTION cnalloc::npy_malloc
#define ARMA_ALIEN_MEM_FREE_FUNCTION cnalloc::npy_free
#ifndef CARMA_ARMA_ALIEN_MEM_FUNCTIONS_SET
  #define CARMA_ARMA_ALIEN_MEM_FUNCTIONS_SET
#endif
#endif  // CNALLOC_INCLUDE_CNALLOC_HPP_
