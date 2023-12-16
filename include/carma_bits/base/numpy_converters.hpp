#pragma once
#include <armadillo>
#include <carma_bits/base/config.hpp>
#include <carma_bits/base/numpy_converters.hpp>
#include <carma_bits/converter_types.hpp>
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
        NumpyContainer container(src);
        FitsArmaType().check<armaT>(container);
        memory_order_policy().template check<armaT>(container);
        if (CARMA_UNLIKELY(container.ill_conditioned || container.order_copy)) {
            carma_debug_print(
                "Using CopyIntoConverter, array ",
                container.arr,
                " does not meet the required conditions and must be copied into the Arma object using Numpy."
            );
            return CopyIntoConverter().get<armaT>(container);
        }
        carma_debug_print(
            "Using CopyConverter, array ", container.arr, " will be copied in by Arma object's constructor."
        );
        return CopyConverter().get<armaT>(container);
    }
};

template <typename armaT>
struct DefaultNumpyConverter {
    template <typename numpyT>
    armaT operator()(numpyT&& src) {
        return internal::NumpyConverter<armaT, CARMA_DEFAULT_MEMORY_ORDER>().template operator(
        )<decltype(src)>(std::forward<numpyT>(src));
    }
};

}  // namespace carma::internal
