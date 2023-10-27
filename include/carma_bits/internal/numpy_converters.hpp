#pragma once

#include <algorithm>
#include <armadillo>
#include <carma_bits/internal/common.hpp>
#include <carma_bits/internal/numpy_container.hpp>
#include <carma_bits/internal/type_traits.hpp>
#include <cstring>
#include <stdexcept>
#include <string>
#include <type_traits>
#include <typeinfo>

namespace carma::internal {

template <typename armaT, iff_Row<armaT> = 0>
inline armaT to_arma(const NumpyContainer& src) {
    using eT = typename armaT::elem_type;
    return arma::Row<eT>(src.data<eT>(), src.n_elem, src.copy_in, src.strict);
}

template <typename armaT, iff_Col<armaT> = 1>
inline armaT to_arma(const NumpyContainer& src) {
    using eT = typename armaT::elem_type;
    return arma::Col<eT>(src.data<eT>(), src.n_elem, src.copy_in, src.strict);
}

template <typename armaT, iff_Mat<armaT> = 2>
inline armaT to_arma(const NumpyContainer& src) {
    using eT = typename armaT::elem_type;
    return arma::Mat<eT>(src.data<eT>(), src.n_rows, src.n_cols, src.copy_in, src.strict);
}

template <typename armaT, iff_Cube<armaT> = 3>
inline armaT to_arma(const NumpyContainer& src) {
    using eT = typename armaT::elem_type;
    return arma::Cube<eT>(src.data<eT>(), src.n_rows, src.n_cols, src.n_slices, src.copy_in, src.strict);
}

// catch against unknown armaT with nicer to understand compile time issue
template <typename armaT, std::enable_if_t<!is_Arma<armaT>::value>>
inline armaT to_arma(const NumpyContainer&) {
    static_assert(!is_Arma<armaT>::value, "|carma| encountered unhandled armaT.");
}

template <typename armaT, iff_Row<armaT> = 0>
inline armaT construct_arma(const NumpyContainer& src) {
    using eT = typename armaT::elem_type;
    return arma::Row<eT>(src.n_elem, arma::fill::none);
}

template <typename armaT, iff_Col<armaT> = 1>
inline armaT construct_arma(const NumpyContainer& src) {
    using eT = typename armaT::elem_type;
    return arma::Col<eT>(src.n_elem, arma::fill::none);
}

template <typename armaT, iff_Mat<armaT> = 2>
inline armaT construct_arma(const NumpyContainer& src) {
    using eT = typename armaT::elem_type;
    return arma::Mat<eT>(src.n_rows, src.n_cols, arma::fill::none);
}

template <typename armaT, iff_Cube<armaT> = 3>
inline armaT construct_arma(const NumpyContainer& src) {
    using eT = typename armaT::elem_type;
    return arma::Cube<eT>(src.n_rows, src.n_cols, src.n_slices, arma::fill::none);
}

// catch against unknown armaT with nicer to understand compile time issue
template <typename armaT, std::enable_if_t<!is_Arma<armaT>::value>>
inline armaT construct_arma(const NumpyContainer&) {
    static_assert(!is_Arma<armaT>::value, "|carma| encountered unhandled armaT.");
}

/**
 * \brief Check if array dimensions are compatible with arma type
 */
class FitsArmaType {
    template <typename armaT, iff_Vec<armaT> = 0>
    inline bool fits(const NumpyContainer& src) {
        return (src.n_dim == 1) || ((src.n_dim == 2) && (src.shape[1] == 1 || src.shape[0] == 1));
    }

    template <typename armaT, iff_Mat<armaT> = 0>
    inline bool fits(const NumpyContainer& src) {
        return (src.n_dim == 2) || ((src.n_dim == 3) && (src.shape[2] == 1 || src.shape[1] == 1 || src.shape[0] == 1));
    }

    template <typename armaT, iff_Cube<armaT> = 0>
    inline bool fits(const NumpyContainer& src) {
        return (src.n_dim == 3)
               || ((src.n_dim == 4)
                   && (src.shape[3] == 1 || src.shape[2] == 1 || src.shape[1] == 1 || src.shape[0] == 1));
    }

   public:
    /**
     * \brief Check if array dimensions are compatible with arma::Row, arma::Col
     *
     * \param[in]   src                 the view of the numpy array
     * \throws      std::runtime_error  if not compatible
     * \return void
     */
    template <typename armaT, iff_Vec<armaT> = 0>
    void check(const NumpyContainer& src) {
        if (CARMA_UNLIKELY((src.n_dim < 1) || (src.n_dim > 2) || (!fits<armaT>(src)))) {
            throw std::runtime_error(
                "|carma| cannot convert array to arma::Vec with dimensions: " + std::to_string(src.n_dim)
            );
        }
    }

    /**
     * \brief Check if array dimensions are compatible with arma::Mat
     *
     * \param[in]   src                the view of the numpy array
     * \throws      std::runtime_error if not compatible
     * \return void
     */
    template <typename armaT, iff_Mat<armaT> = 0>
    void check(const NumpyContainer& src) {
        if (CARMA_UNLIKELY((src.n_dim < 1) || (src.n_dim > 3) || (!fits<armaT>(src)))) {
            throw std::runtime_error(
                "|carma| cannot convert array to arma::Mat with dimensions: " + std::to_string(src.n_dim)
            );
        }
    }

    /**
     * \brief Check if array dimensions are compatible with arma::Cube
     *
     * \param[in]   src                the view of the numpy array
     * \throws      std::runtime_error if not compatible
     * \return void
     */
    template <typename armaT, iff_Cube<armaT> = 0>
    void check(const NumpyContainer& src) {
        if (CARMA_UNLIKELY((src.n_dim < 1) || (src.n_dim > 4) || (!fits<armaT>(src)))) {
            throw std::runtime_error(
                "|carma| cannot convert array to arma::Mat with dimensions: " + std::to_string(src.n_dim)
            );
        }
    }
};

}  // namespace carma::internal
