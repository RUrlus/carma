#pragma once

#include <carma_bits/extension/numpy_container.hpp>
#include <carma_bits/internal/arma_container.hpp>
#include <carma_bits/internal/common.hpp>
#include <carma_bits/internal/numpy_api.hpp>
#include <carma_bits/internal/type_traits.hpp>
#include <utility>

namespace carma {
/* --------------------------------------------------------------
                    Memory order policies
-------------------------------------------------------------- */

/**
 * \brief Memory order policy that looks for C-order contiguous arrays
 *        and transposes them.
 * \details The TransposedRowOrder memory_order_policy expects
 *          that input arrays are row-major/C-order and converts them
 *          to column-major/F-order by transposing the array.
 *          If the array does not have the right order it is marked
 *          to be copied to the right order.
 */
struct TransposedRowOrder {
    template <typename armaT, internal::iff_Arma<armaT> = 0>
    void check(internal::ArmaContainer& src) {
        std::reverse(src.shape.begin(), src.shape.end());
        src.fortran_order = false;
        if constexpr (!internal::is_Vec<armaT>::value) {
            // row and columns are both C and Fortran contiguous
            src.order_flag = py::detail::npy_api::NPY_ARRAY_C_CONTIGUOUS_;
        }
    }

    template <typename armaT, internal::iff_Row<armaT> = 0>
    void check(internal::NumpyContainer& src) {
        src.n_rows = 1;
        src.n_cols = src.n_elem;
    }

    template <typename armaT, internal::iff_Col<armaT> = 0>
    void check(internal::NumpyContainer& src) {
        src.n_rows = src.n_elem;
        src.n_cols = 1;
    }

    template <typename armaT, internal::iff_Mat<armaT> = 0>
    void check(internal::NumpyContainer& src) {
        src.n_rows = src.shape[1];
        src.n_cols = src.shape[0];
        std::swap(src.shape[0], src.shape[1]);
        src.order_copy = src.contiguous != 1;
        src.target_order = NPY_CORDER;
    }

    template <typename armaT, internal::iff_Cube<armaT> = 0>
    void check(internal::NumpyContainer& src) {
        src.n_rows = src.shape[2];
        src.n_cols = src.shape[1];
        src.n_slices = src.shape[0];
        std::reverse(src.shape.begin(), src.shape.end());
        src.order_copy = src.contiguous != 1;
        src.target_order = NPY_CORDER;
    }
#ifdef CARMA_DEBUG
    static constexpr std::string_view name_ = "TransposedRowOrder";
#endif
};

/**
 * \brief Memory order policy that looks for F-order contiguous arrays.
 * \details The ColumnOrder memory_order_policy expects
 *          that input arrays are column-major/F-order.
 *          If the array does not have the right order it is marked
 *          to be copied to the right order.
 */
struct ColumnOrder {
    template <typename armaT, internal::iff_Arma<armaT> = 0>
    void check(internal::ArmaContainer& src) {
        src.fortran_order = true;
    }

    template <typename armaT, internal::iff_Row<armaT> = 0>
    void check(internal::NumpyContainer& src) {
        src.n_rows = 1;
        src.n_cols = src.n_elem;
    }

    template <typename armaT, internal::iff_Col<armaT> = 0>
    void check(internal::NumpyContainer& src) {
        src.n_rows = src.n_elem;
        src.n_cols = 1;
    }
    template <typename armaT, internal::iff_Mat<armaT> = 0>
    void check(internal::NumpyContainer& src) {
        src.n_rows = src.shape[0];
        src.n_cols = src.shape[1];
        src.order_copy = src.contiguous != 2;
        src.target_order = NPY_FORTRANORDER;
    }

    template <typename armaT, internal::iff_Cube<armaT> = 0>
    void check(internal::NumpyContainer& src) {
        src.n_rows = src.shape[0];
        src.n_cols = src.shape[1];
        src.n_slices = src.shape[2];
        src.order_copy = src.contiguous != 2;
        src.target_order = NPY_FORTRANORDER;
    }
#ifdef CARMA_DEBUG
    static constexpr std::string_view name_ = "ColumnOrder";
#endif
};

}  // namespace carma
