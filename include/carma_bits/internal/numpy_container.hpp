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
#include <carma_bits/internal/type_traits.hpp>
#include <cstring>
#include <stdexcept>
#include <string>
#include <utility>

namespace carma::internal {

class NumpyContainer {
   public:
    std::array<py::ssize_t, 4> shape;
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
    explicit NumpyContainer(const py::array_t<eT>& src)
        : obj{src.ptr()},
          arr{reinterpret_cast<PyArrayObject*>(obj)},
          mem{PyArray_DATA(arr)},
          n_elem{static_cast<arma::uword>(src.size())},
          n_dim{static_cast<int>(src.ndim())},
          contiguous{
              is_f_contiguous(arr)   ? 2
              : is_c_contiguous(arr) ? 1
                                     : 0
          },
          owndata{src.owndata()},
          writeable{src.writeable()},
          aligned{is_aligned(arr)},
          ill_conditioned((!aligned) || (!static_cast<bool>(contiguous))) {
        int clipped_n_dim = n_dim < 4 ? n_dim : 4;
        std::memcpy(shape.data(), src.shape(), clipped_n_dim * sizeof(py::ssize_t));
    };

    template <typename eT>
    eT* data() const {
        return static_cast<eT*>(mem);
    }

    /* Use Numpy's api to account for stride, order and copy into the arma object*/
    template <typename armaT, iff_Arma<armaT> = 0>
    inline void copy_into(armaT& dest) {
        using eT = typename armaT::elem_type;
        carma_debug_print("Copying data of array ", obj, " to ", dest.memptr(), " using Numpy.");
        auto api = npy_api::get();
        // make the temporary array writeable and mark the memory as aligned and give the order of the arma object
        int flags
            = (py::detail::npy_api::NPY_ARRAY_ALIGNED_ | py::detail::npy_api::NPY_ARRAY_WRITEABLE_
               | py::detail::npy_api::NPY_ARRAY_F_CONTIGUOUS_);
        // get description from element type
        auto dtype = py::dtype::of<eT>();
        // create Fortran order strides
        auto strides = std::vector<npy_intp>(n_dim, dtype.itemsize());
        for (int i = 1; i < n_dim; ++i) {
            strides[i] = strides[i - 1] * shape[i - 1];
        }
        auto tmp = api.PyArray_NewFromDescr_(
            api.PyArray_Type_, dtype.release().ptr(), n_dim, shape.data(), strides.data(), dest.memptr(), flags, nullptr
        );
        // copy the array to a well behaved target-order
        int ret_code = api.PyArray_CopyInto_(tmp, arr);
        if (ret_code != 0) {
            throw std::runtime_error("|carma| Copy of numpy array failed with ret_code: " + std::to_string(ret_code));
        }

        // make sure to remove owndata flag to prevent memory being freed
        PyArray_CLEARFLAGS(tmp, NPY_ARRAY_OWNDATA);
        // clean up temporary array but not the memory it viewed
        api.PyArray_Free_(tmp, nullptr);
    }  // copy_into

    inline void make_arma_compatible() {
        carma_debug_print("Copying array ", obj, " to Arma compatible layout using Numpy.");
        auto api = npy_api::get();
        PyObject* dest_obj = api.PyArray_NewLikeArray_(arr, NPY_FORTRANORDER, nullptr, 0);
        auto dest_arr = reinterpret_cast<PyArrayObject*>(dest_obj);

        // copy the array to a well behaved target-order
        int ret_code = api.PyArray_CopyInto_(dest_arr, arr);
        if (ret_code != 0) {
            throw std::runtime_error("|carma| Copy of numpy array failed with ret_code: " + std::to_string(ret_code));
        }

        obj = dest_obj;
        arr = dest_arr;
        mem = PyArray_DATA(arr);
        contiguous = 2;
        owndata = true;
        writeable = true;
        aligned = true;
        ill_conditioned = false;
        order_copy = false;
        copy_in = true;
        strict = false;
    }  // make_arma_compatible

    inline void free() {
        carma_extra_debug_print("Freeing array ", arr);
        npy_api::get().PyArray_Free_(arr, mem);
        mem = nullptr;
    }

    // alien methods; defined in carma/alien/array_view.hpp
    void take_ownership();
    template <typename armaT, iff_Arma<armaT>>
    void give_ownership(armaT& dest);
    void steal_copy();
    void swap_copy();
};

}  // namespace carma::internal
