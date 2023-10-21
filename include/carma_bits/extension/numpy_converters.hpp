#pragma once

#include <armadillo>
#include <carma_bits/extension/config.hpp>
#include <carma_bits/extension/converter_types.hpp>
#include <carma_bits/extension/numpy_container.hpp>
#include <carma_bits/internal/common.hpp>
#include <carma_bits/internal/numpy_converters.hpp>
#include <carma_bits/internal/type_traits.hpp>
#include <cstring>
#include <stdexcept>
#include <string>
#include <type_traits>
#include <typeinfo>

namespace carma::internal {

#if defined(CARMA_EXTRA_DEBUG)

template <typename armaT, typename numpyT, typename converter, typename resolution_policy, typename memory_order_policy>
struct NumpyConverterInfo {
    using numpyT_ = numpyT;
    using armaT_ = armaT;
    using converter_ = converter;
    using resolution_ = resolution_policy;
    using mem_order_ = memory_order_policy;
    void operator()(const NumpyContainer& src) {
        std::cout << "\n|----------------------------------------------------------|"
                     "\n"
                  << "|                  CARMA CONVERSION DEBUG                  |"
                  << "\n|----------------------------------------------------------|"
                     "\n|\n";
        std::cout << "| Array address:           " << src.obj << "\n|\n";
        std::cout << "| Conversion configuration:\n"
                  << "| -------------------------\n"
                  << "| * from:                  " << get_full_typename<numpyT_>() << "\n"
                  << "| * to:                    " << get_full_typename<armaT_>() << "\n"
                  << "| * converter:             " << converter_::name_ << "\n"
                  << "| * resolution_policy:     " << resolution_::name_ << "\n"
                  << "| * memory_order_policy:   " << mem_order_::name_ << "\n|\n";

        std::string shape;
        shape.reserve(10);
        shape = "(";
        for (int i = 0; i < src.n_dim; i++) {
            shape += std::to_string(src.shape[i]);
            shape += ",";
        }
        shape += ")";
        std::cout << "| Array attributes:\n"
                  << "| -----------------\n"
                  << "| * data:                  " << src.mem << "\n"
                  << "| * size:                  " << src.n_elem << "\n"
                  << "| * shape:                 " << shape << "\n"
                  << "| * aligned:               " << (src.aligned ? "true" : "false") << "\n"
                  << "| * owndata:               " << (src.owndata ? "true" : "false") << "\n"
                  << "| * writeable:             " << (src.writeable ? "true" : "false") << "\n"
                  << "| * memory order:          "
                  << (src.contiguous == 2   ? "F-order"
                      : src.contiguous == 1 ? "C-order"
                                            : "none")
                  << "\n|\n";

        // needed as memory_order_policy runs after this, we can't move this
        // forward without catching a potential exception regarding fit which we
        // simple avoid here.
        bool order_copy;
        if constexpr (is_ColumnOrder<mem_order_>::value) {
            order_copy = src.contiguous != 2;
        } else {
            order_copy = src.contiguous != 1;
        }

        if constexpr (!is_CopyConverter<converter_>::value) {
            std::cout << "| Copy if:\n"
                      << "| --------\n"
                      << "| * not aligned            [" << (src.aligned ? "false" : "true") << "]\n"
                      << "| * not contiguous         [" << (src.contiguous > 0 ? "false" : "true") << "]\n"
                      << "| * wrong memory order     [" << (order_copy ? "true" : "false") << "]\n";
            if constexpr (is_BorrowConverter<converter_>::value) {
                std::cout << "| * not writeable          [" << (src.writeable ? "false" : "true") << "]\n";
            } else if constexpr (is_MoveConverter<converter_>::value) {
                std::cout << "| * not owndata            [" << (src.owndata ? "false" : "true") << "]\n"
                          << "| * not writeable          [" << (src.writeable ? "false" : "true") << "]\n"
                          << "| * below pre-alloc size   ["
                          << (src.n_elem <= arma::arma_config::mat_prealloc ? "true" : "false") << "]\n";
            }
        }
        std::cout << "|\n|-----------------------------------------------------"
                     "-----|\n\n";
    };
};

#endif  // CARMA_EXTRA_DEBUG

/*
                                NumpyConverter
*/

template <typename armaT, typename converter, typename resolution_policy, typename memory_order_policy>
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
            is_Converter<converter>::value,
            "|carma| `converter` must be one of: BorrowConverter, "
            "CopyConverter, ViewConverter or "
            "MoveConverter."
        );
        static_assert(
            is_ResolutionPolicy<resolution_policy>::value,
            "|carma| `resolution_policy` must be one of: CopyResolution, "
            "RaiseResolution, CopySwapResolution."
        );
        static_assert(
            is_MemoryOrderPolicy<memory_order_policy>::value,
            "|carma| `memory_order_policy` must be one of: ColumnOrder, "
            "TransposedRowOrder."
        );
        static_assert(
            not((is_MoveConverter<converter>::value || is_BorrowConverter<converter>::value)
                && std::is_const_v<std::remove_reference_t<numpyT>>),
            "|carma| BorrowConverter and MoveConverter cannot be used with "
            "`const py::array_t`."
        );
#ifndef CARMA_DONT_ENFORCE_RVALUE_MOVECONVERTER
        static_assert(
            not(is_MoveConverter<converter>::value && (!std::is_rvalue_reference_v<numpyT>)),
            "|carma| [optional] `MoveConverter` is only enabled for r-value "
            "references"
        );
#endif
        NumpyContainer view(src);
#ifdef CARMA_EXTRA_DEBUG
        NumpyConverterInfo<armaT, numpyT, converter, resolution_policy, memory_order_policy>()(view);
#endif  // CARMA_EXTRA_DEBUG
        FitsArmaType().check<armaT>(view);
        memory_order_policy().template check<armaT>(view);
        return resolution_policy().template resolve<armaT, converter>(view);
    }
};

template <
    typename armaT,
    typename resolution_policy = CARMA_DEFAULT_RESOLUTION,
    typename memory_order_policy = CARMA_DEFAULT_MEMORY_ORDER>
struct DefaultNumpyConverter {
    template <typename numpyT>
    armaT operator()(numpyT&& src) {
        if constexpr (std::is_rvalue_reference_v<decltype(src)>) {
            return internal::NumpyConverter<armaT, MoveConverter, resolution_policy, memory_order_policy>()
                .template operator()<decltype(src)>(std::forward<numpyT>(src));
        } else if constexpr (std::is_const_v<std::remove_reference_t<armaT>>) {
            return internal::NumpyConverter<armaT, ViewConverter, resolution_policy, memory_order_policy>()
                .template operator()<decltype(src)>(std::forward<numpyT>(src));
        } else if constexpr (std::is_const_v<std::remove_reference_t<decltype(src)>>) {
            return internal::
                NumpyConverter<armaT, CARMA_DEFAULT_CONST_LVALUE_CONVERTER, resolution_policy, memory_order_policy>()(
                    std::forward<numpyT>(src)
                );
        } else {
            return internal::
                NumpyConverter<armaT, CARMA_DEFAULT_LVALUE_CONVERTER, resolution_policy, memory_order_policy>()(
                    std::forward<numpyT>(src)
                );
        }
    }
};
}  // namespace carma::internal
