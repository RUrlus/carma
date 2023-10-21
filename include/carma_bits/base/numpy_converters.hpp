#pragma once
#include <armadillo>
#include <carma_bits/base/config.hpp>
#include <carma_bits/base/converter_types.hpp>
#include <carma_bits/internal/common.hpp>
#include <carma_bits/internal/numpy_container.hpp>
#include <carma_bits/internal/numpy_converters.hpp>
#include <utility>  // std::forward

namespace carma::internal {

template <typename armaT, iff_Row<armaT> = 0>
inline armaT construct_arma(const NumpyContainer& src) {
    using eT = typename armaT::elem_type;
    return arma::Row<eT>(src.n_elem, arma::fill::none);
};

template <typename armaT, iff_Col<armaT> = 1>
inline armaT construct_arma(const NumpyContainer& src) {
    using eT = typename armaT::elem_type;
    return arma::Col<eT>(src.n_elem, arma::fill::none);
};

template <typename armaT, iff_Mat<armaT> = 2>
inline armaT construct_arma(const NumpyContainer& src) {
    using eT = typename armaT::elem_type;
    return arma::Mat<eT>(src.n_rows, src.n_cols, arma::fill::none);
};

template <typename armaT, iff_Cube<armaT> = 3>
inline armaT construct_arma(const NumpyContainer& src) {
    using eT = typename armaT::elem_type;
    return arma::Cube<eT>(src.n_rows, src.n_cols, src.n_slices, arma::fill::none);
};

// catch against unknown armaT with nicer to understand compile time issue
template <typename armaT, std::enable_if_t<!is_Arma<armaT>::value>>
inline armaT construct_arma(const NumpyContainer&) {
    static_assert(!is_Arma<armaT>::value, "|carma| encountered unhandled armaT.");
};

template <typename armaT, typename memory_order_policy>
struct NumpyConverter {
    template <typename numpyT>
    armaT operator()(numpyT&& src) {
        static_assert(
            is_Numpy<numpyT, typename armaT::elem_type>::value,
            "|carma| `numpyT` must be a specialisation of `py::array_t`."
        );
        static_assert(
            is_Arma<armaT>::value,
            "|carma| `armaT` must be a (subclass of) `arma::Row`, `arma::Col`, "
            "`arma::Mat` or `arma::Cube`."
        );
        static_assert(
            is_MemoryOrderPolicy<memory_order_policy>::value,
            "|carma| `memory_order_policy` must be one of: ColumnOrder, "
            "TransposedRowOrder."
        );
        NumpyContainer view(src);
        FitsArmaType().check<armaT>(view);
        memory_order_policy().template check<armaT>(view);
        if (CARMA_UNLIKELY(src.ill_conditioned || src.order_copy)) {
            return CopyIntoConverter().get<armaT>(src);
        }
        return CopyInConverter().get<armaT>(src);
    }
};

template <typename armaT>
struct DefaultNumpyConverter {
    template <typename numpyT>
    armaT operator()(numpyT&& src) {
        return internal::NumpyConverter<armaT, CARMA_DEFAULT_MEMORY_ORDER>().template operator(
        )<decltype(src)>(std::forward<numpyT>(src));
    };
};

}  // namespace carma::internal
