#ifndef INCLUDE_CARMA_BITS_ARRAY_VIEW_HPP_
#define INCLUDE_CARMA_BITS_ARRAY_VIEW_HPP_

// pybind11 include required even if not explicitly used
// to prevent link with pythonXX_d.lib on Win32
// (cf Py_DEBUG defined in numpy headers and https://github.com/pybind/pybind11/issues/1295)
#include <pybind11/pybind11.h>
// include order matters here
#include <Python.h>
#define NPY_NO_DEPRECATED_API NPY_1_18_API_VERSION
#include <numpy/arrayobject.h>

#include <armadillo>
#include <carma_bits/common.hpp>
#include <carma_bits/numpy_api.hpp>
#include <cstring>
#include <stdexcept>
#include <string>

namespace carma {

namespace internal {

class ArrayView {
   public:
    std::array<ssize_t, 3> shape;
    PyObject* obj;
    PyArrayObject* arr;
    void* mem;
    arma::uword n_elem;
    arma::uword n_rows = 0;
    arma::uword n_cols = 0;
    arma::uword n_slices = 0;
    int n_dim;
    // 0 is non-contigous; 1 is C order; 2 is F order
    int contiguous;
    //-1 for any order; 0 for C-order; 1 for F order
    NPY_ORDER target_order = NPY_ANYORDER;
    bool owndata;
    bool writeable;
    bool aligned;
    bool ill_conditioned;
    bool order_copy = false;
    bool copy_in = false;
    bool strict = true;
    bool stolen_copy = false;

    template <typename eT>
    explicit ArrayView(const py::array_t<eT>& src)
        : obj{src.ptr()},
          arr{reinterpret_cast<PyArrayObject*>(obj)},
          mem{PyArray_DATA(arr)},
          n_elem{static_cast<arma::uword>(src.size())},
          n_dim{static_cast<int>(src.ndim())},
          contiguous{is_f_contiguous(arr)   ? 2
                     : is_c_contiguous(arr) ? 1
                                            : 0},
          owndata{src.owndata()},
          writeable{src.writeable()},
          aligned{is_aligned(arr)} {
        int clipped_n_dim = n_dim < 3 ? n_dim : 3;
        std::memcpy(shape.data(), src.shape(), clipped_n_dim * sizeof(ssize_t));
        ill_conditioned = (!aligned) || (!static_cast<bool>(contiguous));
    };

    template <typename eT>
    eT* data() const {
        return static_cast<eT*>(mem);
    }

    void take_ownership() {
#ifdef CARMA_EXTRA_DEBUG
        std::cout << "|carma| taking ownership of array " << obj << "\n";
#endif
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
    inline void give_ownership(armaT& dest) {
#ifdef CARMA_EXTRA_DEBUG
        std::cout << "|carma| releasing ownership of array " << obj << "to " << (&dest) << "\n";
#endif
        arma::access::rw(dest.n_alloc) = n_elem;
        arma::access::rw(dest.mem_state) = 0;
        release_if_copied_in();
    }

    void release_if_copied_in() {
        if (copy_in) {
#ifdef CARMA_EXTRA_DEBUG
            std::cout << "|carma| array " << obj << " with size " << n_elem
                      << " was copied in, as it does not exceed arma's prealloc size.\n";
#endif
            if (stolen_copy) {
#ifdef CARMA_EXTRA_DEBUG
                std::cout << "|carma| freeing " << mem << "\n";
#endif
                // We copied in because of the array's size in the CopyConverter
                // we need to free the memory as we own it
                npy_api::get().PyDataMem_FREE_(mem);
                mem = nullptr;
            } else {
#ifdef CARMA_EXTRA_DEBUG
                std::cout << "|carma| re-enabling owndata for array " << obj << "\n";
#endif
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
    void steal_copy() {
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
    void swap_copy() {
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
};

}  // namespace internal
}  // namespace carma
#endif  // INCLUDE_CARMA_BITS_ARRAY_VIEW_HPP_
