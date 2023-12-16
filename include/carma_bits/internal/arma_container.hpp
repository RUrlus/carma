#pragma once
#include <algorithm>
#include <armadillo>
#include <carma_bits/internal/type_traits.hpp>
#include <vector>

namespace carma::internal {

class ArmaContainer {
    void* mem;
    py::ssize_t itemsize;
    // aligned and writeable what numpy calls behaved
    int default_flags = py::detail::npy_api::NPY_ARRAY_ALIGNED_ | py::detail::npy_api::NPY_ARRAY_WRITEABLE_;

   public:
    std::vector<py::ssize_t> shape;
    arma::uword n_elem;
    int ndim = 2;
    int order_flag;
    bool writeable = true;
    bool order_copy = false;
    bool copy_out = false;
    bool fortran_order = true;
    void* obj;

    template <typename armaT, typename eT = typename armaT::elem_type, iff_Arma<armaT> = 0>
    explicit ArmaContainer(armaT& src) {
        if constexpr (arma::is_Row<armaT>::value) {
            shape = {1, static_cast<py::ssize_t>(src.n_elem)};
        } else if constexpr (arma::is_Col<armaT>::value) {
            shape = {static_cast<py::ssize_t>(src.n_elem), 1};
            order_flag = py::detail::npy_api::NPY_ARRAY_C_CONTIGUOUS_ | py::detail::npy_api::NPY_ARRAY_F_CONTIGUOUS_;
        } else if constexpr (arma::is_Mat<armaT>::value) {
            shape = {static_cast<py::ssize_t>(src.n_rows), static_cast<py::ssize_t>(src.n_cols)};
        } else if constexpr (arma::is_Cube<armaT>::value) {
            ndim = 3;
            shape
                = {static_cast<py::ssize_t>(src.n_rows),
                   static_cast<py::ssize_t>(src.n_cols),
                   static_cast<py::ssize_t>(src.n_slices)};
        }

        if constexpr (is_Vec<armaT>::value) {
            order_flag = py::detail::npy_api::NPY_ARRAY_C_CONTIGUOUS_ | py::detail::npy_api::NPY_ARRAY_F_CONTIGUOUS_;
        } else {
            order_flag = py::detail::npy_api::NPY_ARRAY_F_CONTIGUOUS_;
        }
        mem = src.memptr();
        itemsize = py::dtype::of<eT>().itemsize();
    }

    [[nodiscard]] int flags() const { return default_flags | order_flag; }

    template <typename eT>
    eT* data() {
        return reinterpret_cast<eT*>(mem);
    }

    [[nodiscard]] std::vector<py::ssize_t> get_shape() const {
        std::vector<py::ssize_t> result(ndim);
        std::copy(shape.begin(), shape.end(), result.begin());
        return result;
    }

    [[nodiscard]] std::vector<py::ssize_t> get_strides() const {
        std::vector<py::ssize_t> result;
        if (fortran_order) {
            result = py::detail::f_strides(shape, itemsize);
        } else {
            result = py::detail::c_strides(shape, itemsize);
        }
        return result;
    }
};
}  // namespace carma::internal
