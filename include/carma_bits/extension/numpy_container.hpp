#pragma once

// pybind11 include required even if not explicitly used
// to prevent link with pythonXX_d.lib on Win32
// (cf Py_DEBUG defined in numpy headers and https://github.com/pybind/pybind11/issues/1295)
#include <pybind11/pybind11.h>
// include order matters here
#include <Python.h>

#define NPY_NO_DEPRECATED_API NPY_1_18_API_VERSION
#include <numpy/arrayobject.h>

#include <armadillo>
#include <carma_bits/internal/common.hpp>
#include <carma_bits/internal/numpy_api.hpp>
#include <carma_bits/internal/numpy_container.hpp>
#include <cstring>
#include <stdexcept>
#include <string>
#include <utility>

namespace carma::internal {

void NumpyContainer::take_ownership() {
    carma_extra_debug_print("taking ownership of array ", obj);
    strict = false;
    copy_in = n_elem <= arma::arma_config::mat_prealloc;
    PyArray_CLEARFLAGS(arr, NPY_ARRAY_OWNDATA);
}

/**
 * \brief Give armadillo object ownership of memory
 *
 * \details Armadillo will free the memory during destruction when the `mem_state == 0` and
 *          when `n_alloc > arma_config::mat_prealloc`.
 *          In cases where the number of elements is below armadillo's pre-allocation limit
 *          the memory will be copied in. This means that we have to free the memory if a copy
 *          of an array was stolen.
 *
 * \param[in]   dest    arma object to be given ownership
 * \return void
 */
template <typename armaT, iff_Arma<armaT> = 0>
inline void NumpyContainer::give_ownership(armaT& dest) {
    carma_extra_debug_print("releasing ownership of array ", obj, " to ", (&dest));
    arma::access::rw(dest.n_alloc) = n_elem;
    arma::access::rw(dest.mem_state) = 0;
    if (copy_in) {
        carma_extra_debug_print(
            "array ", obj, " with size ", n_elem, " was copied in, as it does not exceed arma's prealloc size."
        );
        if (stolen_copy) {
            carma_extra_debug_print("freeing ", mem);
            // We copied in because of the array's size in the CopyConverter
            // we need to free the memory as we own it
            npy_api::get().PyDataMem_FREE_(mem);
            mem = nullptr;
        } else {
            carma_extra_debug_print("re-enabling owndata for array ", obj);
            // We copied in because of the array's size in the MoveConterter
            // if we free the memory any view or array that references this
            // memory will segfault on the python side.
            // We re-enable the owndata flag such that the memory is free'd
            // by Python
            PyArray_ENABLEFLAGS(arr, NPY_ARRAY_OWNDATA);
        }
    }
}

/* Use Numpy's api to account for stride, order and steal the memory */
void NumpyContainer::steal_copy() {
#ifdef CARMA_DEBUG
    void* original_mem = mem;
    std::cout << "|carma| a copy of array " << obj << " will moved into the arma object.\n";
#endif
    auto& api = npy_api::get();
    // build an PyArray to do F-order copy
    auto dest = reinterpret_cast<PyArrayObject*>(api.PyArray_NewLikeArray_(arr, target_order, nullptr, 0));

    // copy the array to a well behaved F-order
    int ret_code = api.PyArray_CopyInto_(dest, arr);
    if (ret_code != 0) {
        throw std::runtime_error("|carma| Copy of array failed with ret_code: " + std::to_string(ret_code));
    }

    mem = PyArray_DATA(dest);
#ifdef CARMA_DEBUG
    std::cout << "|carma| copied data " << original_mem << " to " << mem << "\n";
#endif
    // set OWNDATA to false such that the newly create
    // memory is not freed when the array is cleared
    PyArray_CLEARFLAGS(dest, NPY_ARRAY_OWNDATA);
    // free the array but not the memory
    api.PyArray_Free_(dest, nullptr);
    // ensure that we don't clear the owndata flag from the original array
    stolen_copy = true;
    // arma owns thus not strict
    strict = false;
    // check if an additional copy is needed for arma to take ownership
    copy_in = n_elem <= arma::arma_config::mat_prealloc;
}  // steal_copy_array

/* Use Numpy's api to account for stride, order and copy the new array in place*/
void NumpyContainer::swap_copy() {
#ifdef CARMA_DEBUG
    void* original_mem = mem;
    std::cout << "|carma| array " << obj << " will be copied in-place.\n";
#endif
    auto& api = npy_api::get();
    auto tmp = reinterpret_cast<PyArrayObject*>(api.PyArray_NewLikeArray_(arr, target_order, nullptr, 0));

    // copy the array to a well behaved target-order
    int ret_code = api.PyArray_CopyInto_(tmp, arr);
    if (ret_code != 0) {
        throw std::runtime_error("|carma| Copy of numpy array failed with ret_code: " + std::to_string(ret_code));
    }
    // swap copy into the original array
    auto tmp_of = reinterpret_cast<PyArrayObject_fields*>(tmp);
    auto src_of = reinterpret_cast<PyArrayObject_fields*>(arr);
    std::swap(src_of->data, tmp_of->data);

    // fix strides
    std::swap(src_of->strides, tmp_of->strides);

    PyArray_CLEARFLAGS(arr, NPY_ARRAY_C_CONTIGUOUS | NPY_ARRAY_F_CONTIGUOUS);
    PyArray_ENABLEFLAGS(arr, target_order | NPY_ARRAY_BEHAVED | NPY_ARRAY_OWNDATA);

    // clean up temporary which now contains the old memory
    PyArray_ENABLEFLAGS(tmp, NPY_ARRAY_OWNDATA);

    mem = PyArray_DATA(arr);
#ifdef CARMA_DEBUG
    std::cout << "|carma| copied " << mem << "into " << obj << "in place of " << original_mem << "\n";
    std::cout << "|carma| freeing " << PyArray_DATA(tmp) << "\n";
#endif
    api.PyArray_Free_(tmp, PyArray_DATA(tmp));
}  // swap_copy

}  // namespace carma::internal
