#pragma once

#include <armadillo>
#include <carma_bits/extension/numpy_container.hpp>
#include <carma_bits/internal/arma_container.hpp>
#include <carma_bits/internal/arma_converters.hpp>
#include <carma_bits/internal/common.hpp>
#include <carma_bits/internal/converter_types.hpp>
#include <carma_bits/internal/numpy_converters.hpp>
#include <carma_bits/internal/type_traits.hpp>
#include <cstring>
#include <stdexcept>
#include <string>
#include <type_traits>
#include <typeinfo>

namespace carma {

/* --------------------------------------------------------------
                    Converters
-------------------------------------------------------------- */

/**
 * \brief Borrow the Numpy array's memory
 *
 * \details Convert the Numpy array to `armaT` by borrowing the
 *          memory, aka a mutable view on the memory.
 *          The resulting arma object is strict and does not
 *          own the data.
 *          Borrowing is a good choice when you want to set/change
 *          values but the shape of the object will not change
 *
 *          In order to borrow an array it's memory order should
 *          be:
 *              * writeable
 *              * aligned and contiguous
 *              * compatible with the specified `memory_order_policy`
 *
 * \param[in]   src    the view of the numpy array
 * \return arma object
 */
struct BorrowConverter {
    template <typename armaT, internal::iff_Arma<armaT> = 0>
    armaT get(const internal::NumpyContainer& src) {
        return internal::to_arma<armaT>(src);
    }
#ifdef CARMA_DEBUG
    static constexpr std::string_view name_ = "BorrowConverter";
#endif
};

/**
 * \brief Create const view on the Numpy array's memory
 *
 * \details    Convert the Numpy array to `armaT` by borrowing the
 *             memory, aka a immutable view on the memory.
 *             The resulting arma object is strict and does not
 *             own the data.
 *
 *             In order to create a view, the array's memory order should
 *             be:
 *                 * aligned and contiguous
 *                 * compatible with the specified `memory_order_policy`
 *
 * \param[in]   src    the view of the numpy array
 * \return arma object
 */
struct ViewConverter {
    template <typename armaT, internal::iff_Arma<armaT> = 0>
    armaT get(const internal::NumpyContainer& src) {
        return internal::to_arma<armaT>(src);
    }
#ifdef CARMA_DEBUG
    static constexpr std::string_view name_ = "ViewConverter";
#endif
};

/**
 * \brief Convert by copying the Numpy array's memory
 *
 * \details Convert the Numpy array to `armaT` by copying the
 *          memory. The resulting arma object is _not_ strict
 *          and owns the data.
 *
 *          The copy converter does not have any requirements
 *          with regard to the memory
 *
 * \param[in]   src    the view of the numpy array
 * \return arma object
 */
struct CopyConverter {
    template <typename armaT, internal::iff_Arma<armaT> = 0>
    armaT get(internal::NumpyContainer& src) {
        src.steal_copy();
        auto dest = internal::to_arma<armaT>(src);
        src.give_ownership(dest);
        return dest;
    }

#ifdef CARMA_DEBUG
    static constexpr std::string_view name_ = "CopyConverter";
#endif
};

/**
 * \brief Convert by taking ownership of the Numpy array's memory
 *
 * \details Convert the Numpy array to `armaT` by transfering
 *          ownership of the memory to the armadillo object.
 *          The resulting arma object is _not_ strict
 *          and owns the data.
 *
 *          After conversion the Numpy array will no longer own the
 *          memory, `owndata == false`.
 *
 *          In order to take ownership, the array's memory order should
 *          be:
 *              * owned by the array, aka not a view or alias
 *              * writeable
 *              * aligned and contiguous
 *              * compatible with the specified `memory_order_policy`
 *
 * \param[in]   src    the view of the numpy array
 * \return arma object
 */
struct MoveConverter {
    template <typename armaT, internal::iff_Arma<armaT> = 0>
    armaT get(internal::NumpyContainer& src) {
        src.take_ownership();
        auto dest = internal::to_arma<armaT>(src);
        src.give_ownership(dest);
        return dest;
    }

    template <typename armaT, typename eT = typename armaT::elem_type, internal::iff_Arma<armaT> = 0>
    py::array_t<eT> get(armaT&& src, internal::ArmaContainer& container) {
        container.obj = new armaT(std::move(src));
        return internal::create_owning_array<armaT>(container);
    }
#ifdef CARMA_DEBUG
    static constexpr std::string_view name_ = "MoveConverter";
#endif
};

/* --------------------------------------------------------------
                    Resolution policies
-------------------------------------------------------------- */

namespace internal {

#ifdef CARMA_DEBUG
template <typename converter, bool CopySwapResolution = false>
inline void debug_print_conversion(const internal::NumpyContainer& src) {
    if constexpr (is_MoveConverter<converter>::value) {
        std::cout << "|carma| array " << src.arr << " does not meet MoveConverter conditions\n";
    } else if constexpr (is_ViewConverter<converter>::value) {
        std::cout << "|carma| array " << src.arr << " does not meet ViewConverter conditions\n";
    } else if constexpr (is_BorrowConverter<converter>::value && CopySwapResolution == true) {
        std::cout << "|carma| array " << src.arr << " requires copy-swap to meet BorrowConverter conditions\n";
    }
}
#else
template <typename... T>
inline void debug_print_conversion(const internal::NumpyContainer&){};
#endif

inline std::string get_array_address(const NumpyContainer& src) {
    std::ostringstream stream;
    stream << "|carma| array " << src.arr;
    return stream.str();
}

}  // namespace internal

/**
 * \brief Resolution policy that allows (silent) copying to meet the required
 * conditions when required. \details The CopyResolution is the default
 * resolution policy and will copy the input array when needed and possible.
 * CopyResolution policy cannot resolve when the BorrowConverter is used, the
 * CopySwapResolution policy can handle this scenario.
 */
struct CopyResolution {
    template <typename armaT, typename converter, internal::iff_BorrowConverter<converter> = 0>
    armaT resolve(internal::NumpyContainer& src) {
        if (CARMA_UNLIKELY(src.ill_conditioned || src.order_copy || (!src.writeable))) {
            throw std::runtime_error(
                internal::get_array_address(src) + " does not meet BorrowConverter conditions and would require a copy"
            );
        }
        return BorrowConverter().get<armaT>(src);
    }

    template <typename armaT, typename converter, internal::iff_CopyConverter<converter> = 0>
    armaT resolve(internal::NumpyContainer& src) {
        return CopyConverter().get<armaT>(src);
    }

    template <typename armaT, typename converter, internal::iff_MoveConverter<converter> = 0>
    armaT resolve(internal::NumpyContainer& src) {
        if (CARMA_UNLIKELY(src.ill_conditioned || src.order_copy || (!src.owndata) || (!src.writeable))) {
            internal::debug_print_conversion<converter>(src);
            return CopyConverter().get<armaT>(src);
        }
        return MoveConverter().get<armaT>(src);
    }

    template <typename armaT, typename converter, internal::iff_ViewConverter<converter> = 0>
    armaT resolve(internal::NumpyContainer& src) {
        if (CARMA_UNLIKELY(src.ill_conditioned || src.order_copy)) {
            internal::debug_print_conversion<converter>(src);
            return CopyConverter().get<armaT>(src);
        }
        return ViewConverter().get<armaT>(src);
    }
#ifdef CARMA_DEBUG
    static constexpr std::string_view name_ = "CopyResolution";
#endif
};

/**
 * \brief Resolution policy that raises an runtime exception when the required
 * conditions are not met. \details The RaiseResolution is the strictest policy
 * and will raise an exception if any condition is not met, in contrast the
 * CopyResolution will silently copy when it needs and can. This policy should
 * be used when silent copies are undesired or prohibitively expensive.
 */
struct RaiseResolution {
    template <typename armaT, typename converter, internal::iff_BorrowConverter<converter> = 0>
    armaT resolve(internal::NumpyContainer& src) {
        if (CARMA_UNLIKELY(src.ill_conditioned || src.order_copy || (!src.writeable))) {
            throw std::runtime_error(
                internal::get_array_address(src) + " does not meet BorrowConverter conditions and would require a copy"
            );
        }
        return BorrowConverter().get<armaT>(src);
    }

    template <typename armaT, typename converter, internal::iff_CopyConverter<converter> = 0>
    armaT resolve(internal::NumpyContainer& src) {
        return CopyConverter().get<armaT>(src);
    }

    template <typename armaT, typename converter, internal::iff_MoveConverter<converter> = 0>
    armaT resolve(internal::NumpyContainer& src) {
        if (CARMA_UNLIKELY(src.ill_conditioned || src.order_copy || (!src.owndata) || (!src.writeable))) {
            throw std::runtime_error(
                internal::get_array_address(src) + " does not meet MoveConverter conditions and would require a copy"
            );
        }
        return MoveConverter().get<armaT>(src);
    }

    template <typename armaT, typename converter, internal::iff_ViewConverter<converter> = 0>
    armaT resolve(internal::NumpyContainer& src) {
        if (CARMA_UNLIKELY(src.ill_conditioned || src.order_copy)) {
            throw std::runtime_error(
                internal::get_array_address(src) + " does not meet ViewConverter conditions and would require a copy"
            );
        }
        return ViewConverter().get<armaT>(src);
    }
#ifdef CARMA_DEBUG
    static constexpr std::string_view name_ = "RaiseResolution";
#endif
};

/**
 * \brief Resolution policy that allows (silent) copying to meet the required
 * conditions when required even with BorrowConverter.
 *
 * \details The CopySwapResolution is behaves identically to CopyResolution policy with the
 * exception that it can handle ill conditioned and/or arrays with the wrong
 * memory layout. An exception is raised when the array does not own it's memory
 * or is marked as not writeable.
 *
 * \warning CopySwapResolution handles ill conditioned memory by copying the
 * array's memory to the right state and swapping it in the place of the existing memory.
 * This makes use of an deprecated numpy function to directly interface with the array fields. As
 * such this resolution policy should be considered experimental. This policy
 * will likely not work with Numpy >= v2.0
 */
struct CopySwapResolution {
    template <typename armaT, typename converter, internal::iff_BorrowConverter<converter> = 0>
    armaT resolve(internal::NumpyContainer& src) {
        if (CARMA_UNLIKELY((!src.writeable) || (!src.owndata))) {
            throw std::runtime_error(
                internal::get_array_address(src)
                + " cannot copy-swapped as it does not own the data or is not writeable"
            );
        } else if (CARMA_UNLIKELY(src.ill_conditioned || src.order_copy)) {
            internal::debug_print_conversion<converter, true>(src);
            src.swap_copy();
        }
        return BorrowConverter().get<armaT>(src);
    }

    template <typename armaT, typename converter, internal::iff_CopyConverter<converter> = 0>
    armaT resolve(internal::NumpyContainer& src) {
        return CopyConverter().get<armaT>(src);
    }

    template <typename armaT, typename converter, internal::iff_MoveConverter<converter> = 0>
    armaT resolve(internal::NumpyContainer& src) {
        if (CARMA_UNLIKELY(src.ill_conditioned || src.order_copy || (!src.owndata) || (!src.writeable))) {
            internal::debug_print_conversion<converter>(src);
            return CopyConverter().get<armaT>(src);
        }
        return MoveConverter().get<armaT>(src);
    }

    template <typename armaT, typename converter, internal::iff_ViewConverter<converter> = 0>
    armaT resolve(internal::NumpyContainer& src) {
        if (CARMA_UNLIKELY(src.ill_conditioned || src.order_copy)) {
            internal::debug_print_conversion<converter>(src);
            return CopyConverter().get<armaT>(src);
        }
        return ViewConverter().get<armaT>(src);
    }
#ifdef CARMA_DEBUG
    static constexpr std::string_view name_ = "CopySwapResolution";
#endif
};
}  // namespace carma
