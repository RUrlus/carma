#ifndef INCLUDE_CARMA_BITS_TO_ARMA_HPP_
#define INCLUDE_CARMA_BITS_TO_ARMA_HPP_

#include <armadillo>
#include <carma_bits/array_view.hpp>
#include <carma_bits/common.hpp>
#include <cstring>
#include <stdexcept>
#include <string>
#include <type_traits>
#include <typeinfo>

namespace carma {

namespace internal {

template <typename armaT, iff_Row<armaT> = 0>
inline armaT to_arma(const ArrayView& src) {
    using eT = typename armaT::elem_type;
    return arma::Row<eT>(src.data<eT>(), src.n_elem, src.copy_in, src.strict);
};

template <typename armaT, iff_Col<armaT> = 1>
inline armaT to_arma(const ArrayView& src) {
    using eT = typename armaT::elem_type;
    return arma::Col<eT>(src.data<eT>(), src.n_elem, src.copy_in, src.strict);
};

template <typename armaT, iff_Mat<armaT> = 2>
inline armaT to_arma(const ArrayView& src) {
    using eT = typename armaT::elem_type;
    return arma::Mat<eT>(src.data<eT>(), src.n_rows, src.n_cols, src.copy_in, src.strict);
};

template <typename armaT, iff_Cube<armaT> = 3>
inline armaT to_arma(const ArrayView& src) {
    using eT = typename armaT::elem_type;
    return arma::Cube<eT>(src.data<eT>(), src.n_rows, src.n_cols, src.n_slices, src.copy_in, src.strict);
};

// catch against unknown armaT with nicer to understand compile time issue
template <typename armaT, std::enable_if_t<!is_Arma<armaT>::value>>
inline armaT to_arma(const ArrayView&) {
    static_assert(!is_Arma<armaT>::value, "|carma| encountered unhandled armaT.");
};

/**
 * \brief Check if array dimensions are compatible with arma type
 */
class FitsArmaType {
    template <typename armaT, iff_Vec<armaT> = 0>
    inline bool fits(const ArrayView& src) {
        return (src.n_dim == 1) || ((src.n_dim == 2) && (src.shape[1] == 1 || src.shape[0] == 1));
    }

    template <typename armaT, iff_Mat<armaT> = 0>
    inline bool fits(const ArrayView& src) {
        return (src.n_dim == 2) || ((src.n_dim == 3) && (src.shape[2] == 1 || src.shape[1] == 1 || src.shape[0] == 1));
    }

    template <typename armaT, iff_Cube<armaT> = 0>
    inline bool fits(const ArrayView& src) {
        return (src.n_dim == 3) ||
               ((src.n_dim == 4) && (src.shape[3] == 1 || src.shape[2] == 1 || src.shape[1] == 1 || src.shape[0] == 1));
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
    void check(const ArrayView& src) {
        if (CARMA_UNLIKELY((src.n_dim < 1) || (src.n_dim > 2) || (!fits<armaT>(src)))) {
            throw std::runtime_error("|carma| cannot convert array to arma::Vec with dimensions: " +
                                     std::to_string(src.n_dim));
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
    void check(const ArrayView& src) {
        if (CARMA_UNLIKELY((src.n_dim < 1) || (src.n_dim > 3) || (!fits<armaT>(src)))) {
            throw std::runtime_error("|carma| cannot convert array to arma::Mat with dimensions: " +
                                     std::to_string(src.n_dim));
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
    void check(const ArrayView& src) {
        if (CARMA_UNLIKELY((src.n_dim < 1) || (src.n_dim > 4) || (!fits<armaT>(src)))) {
            throw std::runtime_error("|carma| cannot convert array to arma::Mat with dimensions: " +
                                     std::to_string(src.n_dim));
        }
    }
};

}  // namespace internal

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
    template <typename armaT, iff_Arma<armaT> = 0>
    armaT get(const internal::ArrayView& src) {
        return internal::to_arma<armaT>(src);
    };
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
    template <typename armaT, iff_Arma<armaT> = 0>
    armaT get(const internal::ArrayView& src) {
        return internal::to_arma<armaT>(src);
    };
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
    template <typename armaT, iff_Arma<armaT> = 0>
    armaT get(internal::ArrayView& src) {
        src.steal_copy();
        auto dest = internal::to_arma<armaT>(src);
        src.give_ownership(dest);
        return dest;
    };

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
    template <typename armaT, iff_Arma<armaT> = 0>
    armaT get(internal::ArrayView& src) {
        src.take_ownership();
        auto dest = internal::to_arma<armaT>(src);
        src.give_ownership(dest);
        return dest;
    };
#ifdef CARMA_DEBUG
    static constexpr std::string_view name_ = "MoveConverter";
#endif
};

template <typename T>
using is_BorrowConverter = std::is_same<T, BorrowConverter>;

template <typename T>
using is_CopyConverter = std::is_same<T, CopyConverter>;

template <typename T>
using is_MoveConverter = std::is_same<T, MoveConverter>;

template <typename T>
using is_ViewConverter = std::is_same<T, ViewConverter>;

template <typename T>
struct is_Converter {
    static constexpr bool value = (is_BorrowConverter<T>::value || is_CopyConverter<T>::value ||
                                   is_ViewConverter<T>::value || is_MoveConverter<T>::value);
};

template <typename T>
using iff_BorrowConverter = std::enable_if_t<is_BorrowConverter<T>::value, int>;

template <typename T>
using iff_CopyConverter = std::enable_if_t<is_CopyConverter<T>::value, int>;

template <typename T>
using iff_MoveConverter = std::enable_if_t<is_MoveConverter<T>::value, int>;

template <typename T>
using iff_ViewConverter = std::enable_if_t<is_ViewConverter<T>::value, int>;

template <typename T>
using iff_Borrow_or_CopyConverter = std::enable_if_t<is_BorrowConverter<T>::value || is_CopyConverter<T>::value, int>;

template <typename T>
using iff_Converter = std::enable_if_t<is_BorrowConverter<T>::value || is_CopyConverter<T>::value ||
                                           is_MoveConverter<T>::value || is_ViewConverter<T>::value,
                                       int>;

template <typename T>
using iff_mutable_Converter =
    std::enable_if_t<is_BorrowConverter<T>::value || is_CopyConverter<T>::value || is_MoveConverter<T>::value, int>;

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
    template <typename aramT, iff_Row<aramT> = 0>
    void check(internal::ArrayView& src) {
        src.n_rows = 1;
        src.n_cols = src.n_elem;
    };

    template <typename aramT, iff_Col<aramT> = 0>
    void check(internal::ArrayView& src) {
        src.n_rows = src.n_elem;
        src.n_cols = 1;
    };

    template <typename aramT, iff_Mat<aramT> = 0>
    void check(internal::ArrayView& src) {
        src.n_rows = src.shape[1];
        src.n_cols = src.shape[0];
        src.order_copy = src.contiguous != 1;
        src.target_order = NPY_CORDER;
    };

    template <typename aramT, iff_Cube<aramT> = 0>
    void check(internal::ArrayView& src) {
        src.n_rows = src.shape[2];
        src.n_cols = src.shape[1];
        src.n_slices = src.shape[0];
        src.order_copy = src.contiguous != 1;
        src.target_order = NPY_CORDER;
    };
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
    template <typename aramT, iff_Row<aramT> = 0>
    void check(internal::ArrayView& src) {
        src.n_rows = 1;
        src.n_cols = src.n_elem;
    };

    template <typename aramT, iff_Col<aramT> = 0>
    void check(internal::ArrayView& src) {
        src.n_rows = src.n_elem;
        src.n_cols = 1;
    };
    template <typename aramT, iff_Mat<aramT> = 0>
    void check(internal::ArrayView& src) {
        src.n_rows = src.shape[0];
        src.n_cols = src.shape[1];
        src.order_copy = src.contiguous != 2;
        src.target_order = NPY_FORTRANORDER;
    };

    template <typename aramT, iff_Cube<aramT> = 0>
    void check(internal::ArrayView& src) {
        src.n_rows = src.shape[0];
        src.n_cols = src.shape[1];
        src.n_slices = src.shape[2];
        src.order_copy = src.contiguous != 2;
        src.target_order = NPY_FORTRANORDER;
    };
#ifdef CARMA_DEBUG
    static constexpr std::string_view name_ = "ColumnOrder";
#endif
};

template <typename T>
struct is_ColumnOrder {
    static constexpr bool value = std::is_same_v<T, ColumnOrder>;
};

template <typename T>
struct is_TransposedRowOrder {
    static constexpr bool value = std::is_same_v<T, TransposedRowOrder>;
};

template <typename T>
struct is_MemoryOrderPolicy {
    static constexpr bool value = (std::is_same_v<T, ColumnOrder> || std::is_same_v<T, TransposedRowOrder>);
};

/* --------------------------------------------------------------
                    Resolution policies
-------------------------------------------------------------- */

namespace internal {

#ifdef CARMA_DEBUG
template <typename converter, bool CopySwapResolution = false>
inline void carma_debug_print(const internal::ArrayView& src) {
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
inline void carma_debug_print(const internal::ArrayView&){};
#endif

inline std::string get_array_address(const ArrayView& src) {
    std::ostringstream stream;
    stream << "|carma| array " << src.arr;
    return stream.str();
}

}  // namespace internal

/**
 * \brief Resolution policy that allows (silent) copying to meet the required conditions when required.
 * \details The CopyResolution is the default resolution policy and will copy the input array when
 *          needed and possible. CopyResolution policy cannot resolve when the BorrowConverter is used,
 *          the CopySwapResolution policy can handle this scenario.
 */
struct CopyResolution {
    template <typename armaT, typename converter, iff_BorrowConverter<converter> = 0>
    armaT resolve(internal::ArrayView& src) {
        if (CARMA_UNLIKELY(src.ill_conditioned || src.order_copy || (!src.writeable))) {
            throw std::runtime_error(internal::get_array_address(src) +
                                     " does not meet BorrowConverter conditions and would require a copy");
        }
        return BorrowConverter().get<armaT>(src);
    }

    template <typename armaT, typename converter, iff_CopyConverter<converter> = 0>
    armaT resolve(internal::ArrayView& src) {
        return CopyConverter().get<armaT>(src);
    }

    template <typename armaT, typename converter, iff_MoveConverter<converter> = 0>
    armaT resolve(internal::ArrayView& src) {
        if (CARMA_UNLIKELY(src.ill_conditioned || src.order_copy || (!src.owndata) || (!src.writeable))) {
            internal::carma_debug_print<converter>(src);
            return CopyConverter().get<armaT>(src);
        }
        return MoveConverter().get<armaT>(src);
    }

    template <typename armaT, typename converter, iff_ViewConverter<converter> = 0>
    armaT resolve(internal::ArrayView& src) {
        if (CARMA_UNLIKELY(src.ill_conditioned || src.order_copy)) {
            internal::carma_debug_print<converter>(src);
            return CopyConverter().get<armaT>(src);
        }
        return ViewConverter().get<armaT>(src);
    };
#ifdef CARMA_DEBUG
    static constexpr std::string_view name_ = "CopyResolution";
#endif
};

/**
 * \brief Resolution policy that raises an runtime exception when the required conditions are not met.
 * \details The RaiseResolution is the strictest policy and will raise an exception if any condition is
 *          not met, in contrast the CopyResolution will silently copy when it needs and can.
 *          This policy should be used when silent copies are undesired or prohibitively expensive.
 */
struct RaiseResolution {
    template <typename armaT, typename converter, iff_BorrowConverter<converter> = 0>
    armaT resolve(internal::ArrayView& src) {
        if (CARMA_UNLIKELY(src.ill_conditioned || src.order_copy || (!src.writeable))) {
            throw std::runtime_error(internal::get_array_address(src) +
                                     " does not meet BorrowConverter conditions and would require a copy");
        }
        return BorrowConverter().get<armaT>(src);
    }

    template <typename armaT, typename converter, iff_CopyConverter<converter> = 0>
    armaT resolve(internal::ArrayView& src) {
        return CopyConverter().get<armaT>(src);
    }

    template <typename armaT, typename converter, iff_MoveConverter<converter> = 0>
    armaT resolve(internal::ArrayView& src) {
        if (CARMA_UNLIKELY(src.ill_conditioned || src.order_copy || (!src.owndata) || (!src.writeable))) {
            throw std::runtime_error(internal::get_array_address(src) +
                                     " does not meet MoveConverter conditions and would require a copy");
        }
        return MoveConverter().get<armaT>(src);
    }

    template <typename armaT, typename converter, iff_ViewConverter<converter> = 0>
    armaT resolve(internal::ArrayView& src) {
        if (CARMA_UNLIKELY(src.ill_conditioned || src.order_copy)) {
            throw std::runtime_error(internal::get_array_address(src) +
                                     " does not meet ViewConverter conditions and would require a copy");
        }
        return ViewConverter().get<armaT>(src);
    };
#ifdef CARMA_DEBUG
    static constexpr std::string_view name_ = "RaiseResolution";
#endif
};

/**
 * \brief Resolution policy that allows (silent) copying to meet the required conditions when required even with
 *        BorrowConverter.
 * \details The CopySwapResolution is behaves identically to CopyResolution policy with the exception
 *          that it can handle ill conditioned and/or arrays with the wrong memory layout.
 *          An exception is raised when the array does not own it's memory or is marked as not writeable.
 * \warning CopySwapResolution handles ill conditioned memory by copying the array's memory to the right state
 *          and swapping it in the place of the existing memory. This makes use of an deprecated numpy function
 *          to directly interface with the array fields. As such this resolution policy should be considered
 *          experimental. This policy will likely not work with Numpy >= v2.0
 */
struct CopySwapResolution {
    template <typename armaT, typename converter, iff_BorrowConverter<converter> = 0>
    armaT resolve(internal::ArrayView& src) {
        if (CARMA_UNLIKELY((!src.writeable) || (!src.owndata))) {
            throw std::runtime_error(internal::get_array_address(src) +
                                     " cannot copy-swapped as it does not own the data or is not writeable");
        } else if (CARMA_UNLIKELY(src.ill_conditioned || src.order_copy)) {
            internal::carma_debug_print<converter, true>(src);
            src.swap_copy();
        }
        return BorrowConverter().get<armaT>(src);
    }

    template <typename armaT, typename converter, iff_CopyConverter<converter> = 0>
    armaT resolve(internal::ArrayView& src) {
        return CopyConverter().get<armaT>(src);
    }

    template <typename armaT, typename converter, iff_MoveConverter<converter> = 0>
    armaT resolve(internal::ArrayView& src) {
        if (CARMA_UNLIKELY(src.ill_conditioned || src.order_copy || (!src.owndata) || (!src.writeable))) {
            internal::carma_debug_print<converter>(src);
            return CopyConverter().get<armaT>(src);
        }
        return MoveConverter().get<armaT>(src);
    }

    template <typename armaT, typename converter, iff_ViewConverter<converter> = 0>
    armaT resolve(internal::ArrayView& src) {
        if (CARMA_UNLIKELY(src.ill_conditioned || src.order_copy)) {
            internal::carma_debug_print<converter>(src);
            return CopyConverter().get<armaT>(src);
        }
        return ViewConverter().get<armaT>(src);
    };
#ifdef CARMA_DEBUG
    static constexpr std::string_view name_ = "CopySwapResolution";
#endif
};

template <typename T>
struct is_RaiseResolution {
    static constexpr bool value = std::is_same_v<T, RaiseResolution>;
};

template <typename T>
struct is_CopyResolution {
    static constexpr bool value = std::is_same_v<T, CopyResolution>;
};

template <typename T>
struct is_CopySwapResolution {
    static constexpr bool value = std::is_same_v<T, CopySwapResolution>;
};

template <typename T>
struct is_ResolutionPolicy {
    static constexpr bool value = (std::is_same_v<T, CopyResolution> || std::is_same_v<T, RaiseResolution> ||
                                   std::is_same_v<T, CopySwapResolution>);
};

/* --------------------------------------------------------------
                    ConversionConfig
-------------------------------------------------------------- */
#ifndef CARMA_DEFAULT_LVALUE_CONVERTER
#define CARMA_DEFAULT_LVALUE_CONVERTER carma::BorrowConverter
#endif  // CARMA_DEFAULT_LVALUE_CONVERTER

#ifndef CARMA_DEFAULT_CONST_LVALUE_CONVERTER
#define CARMA_DEFAULT_CONST_LVALUE_CONVERTER carma::CopyConverter
#endif  // CARMA_DEFAULT_CONST_LVALUE_CONVERTER

#ifndef CARMA_DEFAULT_RESOLUTION
#define CARMA_DEFAULT_RESOLUTION carma::CopyResolution
#endif  // CARMA_DEFAULT_RESOLUTION

#ifndef CARMA_DEFAULT_MEMORY_ORDER
#define CARMA_DEFAULT_MEMORY_ORDER carma::ColumnOrder
#endif  // CARMA_DEFAULT_MEMORY_ORDER

/**
 * \brief Create compile-time configuration object for Numpy to Armadillo conversion.
 *
 * \tparam converter the converter to be used options are: BorrowConverter, CopyConverter, MoveConverter,
 *                   ViewConverter
 * \tparam resolution_policy which resolution policy to use when the array cannot be converted directly,
 *                                            options are: RaiseResolution, CopyResolution, CopySwapResolution
 * \tparam memory_order_policy which memory order policy to use, options are: ColumnOrder, TransposedRowOrder
 */
template <class converter, class resolution_policy = CARMA_DEFAULT_RESOLUTION,
          class memory_order_policy = CARMA_DEFAULT_MEMORY_ORDER>
struct ConversionConfig {
    static_assert(
        is_Converter<converter>::value,
        "|carma| `converter` must be one of: BorrowConverter, CopyConverter, ViewConverter or MoveConverter.");
    using converter_ = converter;
    static_assert(is_ResolutionPolicy<resolution_policy>::value,
                  "|carma| `resolution_policy` must be one of: CopyResolution, RaiseResolution, CopySwapResolution.");
    using resolution_ = resolution_policy;
    static_assert(is_MemoryOrderPolicy<memory_order_policy>::value,
                  "|carma| `memory_order_policy` must be one of: ColumnOrder, TransposedRowOrder.");
    using mem_order_ = memory_order_policy;
};

/*

                                ConversionConfig type_traits

*/
template <typename T>
using is_ConversionConfig = internal::is_instance<T, ConversionConfig>;

template <typename T>
using iff_ConversionConfig = std::enable_if_t<is_ConversionConfig<T>::value, int>;

namespace internal {

#if defined(CARMA_EXTRA_DEBUG)

template <typename armaT, typename numpyT, typename converter, typename resolution_policy, typename memory_order_policy>
struct npConverterInfo {
    using numpyT_ = numpyT;
    using armaT_ = armaT;
    using converter_ = converter;
    using resolution_ = resolution_policy;
    using mem_order_ = memory_order_policy;
    void operator()(const ArrayView& src) {
        std::cout << "\n|----------------------------------------------------------|\n"
                  << "|                  CARMA CONVERSION DEBUG                  |"
                  << "\n|----------------------------------------------------------|\n|\n";
        std::cout << "| Array address:           " << src.obj << "\n|\n";
        std::cout << "| Conversion configuration:\n"
                  << "| -------------------------\n"
                  << "| * from:                  " << get_full_typename<numpyT_>() << "\n"
                  << "| * to:                    " << get_full_typename<armaT_>() << "\n"
                  << "| * converter:             " << converter_::name_ << "\n"
                  << "| * resolution_policy:     " << resolution_::name_ << "\n"
                  << "| * memory_order_policy:   " << mem_order_::name_ << "\n|\n";

        std::string shape;
        shape.reserve(8);
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

        // needed as memory_order_policy runs after this, we can't move this forward without
        // catching a potential exception regarding fit which we simple avoid here.
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
        std::cout << "|\n|----------------------------------------------------------|\n\n";
    };
};

#endif  // CARMA_EXTRA_DEBUG

/*

                                npConverterImpl

*/

template <typename armaT, typename converter, typename resolution_policy, typename memory_order_policy>
struct npConverterImpl {
    template <typename numpyT>
    armaT operator()(numpyT&& src) {
        static_assert(is_Numpy<numpyT, typename armaT::elem_type>::value,
                      "|carma| `numpyT` must be a specialisation of `py::array_t`.");
        static_assert(is_Arma<armaT>::value,
                      "|carma| `armaT` must be a (subclass of) `arma::Row`, `arma::Col`, `arma::Mat` or `arma::Cube`.");
        static_assert(not((is_MoveConverter<converter>::value || is_BorrowConverter<converter>::value) &&
                          std::is_const_v<std::remove_reference_t<numpyT>>),
                      "|carma| BorrowConverter and MoveConverter cannot be used with `const py::array_t`.");
#ifndef CARMA_DONT_ENFORCE_RVALUE_MOVECONVERTER
        static_assert(not(is_MoveConverter<converter>::value && (!std::is_rvalue_reference_v<numpyT>)),
                      "|carma| [optional] `MoveConverter` is only enabled for r-value references");
#endif
        ArrayView view(src);
#ifdef CARMA_EXTRA_DEBUG
        npConverterInfo<armaT, numpyT, converter, resolution_policy, memory_order_policy>()(view);
#endif  // CARMA_EXTRA_DEBUG
        FitsArmaType().check<armaT>(view);
        memory_order_policy().template check<armaT>(view);
        return resolution_policy().template resolve<armaT, converter>(view);
    }
};

/*

                                npConverterBase

*/
template <typename armaT, typename numpyT, typename converter, typename resolution_policy = CARMA_DEFAULT_RESOLUTION,
          typename memory_order_policy = CARMA_DEFAULT_MEMORY_ORDER>
struct npConverterBase {
    armaT operator()(numpyT&& src) {
        // check template arguments
        static_assert(is_Converter<converter>::value,
                      "|carma| `converter` must be one of: BorrowConverter, CopyConverter, ViewConverter or "
                      "MoveConverter.");
        static_assert(
            is_ResolutionPolicy<resolution_policy>::value,
            "|carma| `resolution_policy` must be one of: CopyResolution, RaiseResolution, CopySwapResolution.");
        static_assert(is_MemoryOrderPolicy<memory_order_policy>::value,
                      "|carma| `memory_order_policy` must be one of: ColumnOrder, TransposedRowOrder.");
        return internal::npConverterImpl<armaT, converter, resolution_policy, memory_order_policy>()(
            std::forward<numpyT>(src));
    }
};

template <typename armaT, typename resolution_policy = CARMA_DEFAULT_RESOLUTION,
          typename memory_order_policy = CARMA_DEFAULT_MEMORY_ORDER>
struct toArma {
    template <typename numpyT>
    armaT operator()(numpyT&& src) {
        if constexpr (std::is_rvalue_reference_v<decltype(src)>) {
            return internal::npConverterImpl<armaT, MoveConverter, resolution_policy, memory_order_policy>()
                .template operator()<decltype(src)>(std::forward<numpyT>(src));
        } else if constexpr (std::is_const_v<std::remove_reference_t<armaT>>) {
            return internal::npConverterImpl<armaT, ViewConverter, resolution_policy, memory_order_policy>()(
                std::forward<numpyT>(src));
        } else if constexpr (std::is_const_v<std::remove_reference_t<decltype(src)>>) {
            return internal::npConverterImpl<armaT, CARMA_DEFAULT_CONST_LVALUE_CONVERTER, resolution_policy,
                                             memory_order_policy>()(std::forward<numpyT>(src));
        } else {
            return internal::npConverterImpl<armaT, CARMA_DEFAULT_LVALUE_CONVERTER, resolution_policy,
                                             memory_order_policy>()(std::forward<numpyT>(src));
        }
    }
};

}  // namespace internal

}  // namespace carma
#endif  // INCLUDE_CARMA_BITS_TO_ARMA_HPP_
