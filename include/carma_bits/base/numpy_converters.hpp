#pragma once
#include <armadillo>
#include <carma_bits/base/config.hpp>
#include <carma_bits/base/converter_types.hpp>
#include <carma_bits/internal/common.hpp>
#include <carma_bits/internal/numpy_container.hpp>
#include <carma_bits/internal/numpy_converters.hpp>
#include <utility>  // std::forward

namespace carma::internal {

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
