#pragma once

#include <pybind11/numpy.h>
#include <pybind11/pybind11.h>

#include <armadillo>
#include <carma_bits/converter_types.hpp>
#include <type_traits>
#include <typeinfo>

namespace carma {

namespace py = pybind11;

namespace internal {

template <typename, template <typename...> typename>
// struct is_instance_impl : public std::false_type {};
struct is_instance_impl {
    static constexpr bool value = false;
};

template <template <typename...> typename U, typename... Ts>
struct is_instance_impl<U<Ts...>, U> {
    static constexpr bool value = true;
};

template <typename T, template <typename...> typename U>
// using is_instance = is_instance_impl<std::decay_t<T>, U>;
struct is_instance {
    static constexpr bool value = is_instance_impl<std::decay_t<T>, U>::value;
};

template <typename targetType, typename refType>
struct is_same_stripped {
    static constexpr bool value
        = (std::is_same_v<std::remove_cv_t<std::remove_reference_t<targetType>>, refType>
           || std::is_same_v<std::remove_cv_t<std::remove_reference_t<targetType>>, refType*>);
};

template <typename T>
using iff_const = std::enable_if_t<std::is_const_v<T>, int>;

template <typename numpyT, typename eT>
struct is_Numpy {
    static constexpr bool value = internal::is_same_stripped<numpyT, py::array_t<eT>>::value;
};

template <typename numpyT, typename eT>
using iff_Numpy = std::enable_if_t<is_Numpy<numpyT, eT>::value, int>;

template <typename T>
using get_baseT = std::remove_cv_t<std::remove_reference_t<T>>;

template <typename armaT>
using armaT_eT = typename get_baseT<armaT>::elem_type;

// FIXME
// Use Arma's approach
// template <typename armaT, typename eT>
// struct is_armaMat_only {
//     static constexpr bool value = internal::is_same_stripped<armaT,
//     arma::MatFixed>::value;
// };

template <typename armaT, typename baseT = get_baseT<armaT>>
using iff_Row = std::enable_if_t<arma::is_Row<baseT>::value, int>;

template <typename armaT, typename baseT = get_baseT<armaT>>
using iff_Col = std::enable_if_t<arma::is_Col<baseT>::value, int>;

template <typename armaT, typename baseT = get_baseT<armaT>>
using iff_Vec = std::enable_if_t<arma::is_Col<baseT>::value || arma::is_Row<baseT>::value, int>;

template <typename armaT, typename baseT = get_baseT<armaT>>
using iff_Mat = std::enable_if_t<arma::is_Mat_only<baseT>::value, int>;

template <typename armaT, typename baseT = get_baseT<armaT>>
using iff_MatOrVec = std::enable_if_t<arma::is_Mat<baseT>::value, int>;

template <typename armaT, typename baseT = get_baseT<armaT>>
using iff_Cube = std::enable_if_t<arma::is_Cube<baseT>::value, int>;

template <typename armaT, typename baseT = get_baseT<armaT>>
using iff_Arma = std::enable_if_t<arma::is_Mat<baseT>::value || arma::is_Cube<baseT>::value, int>;

template <typename armaT>
struct is_Vec {
    static constexpr bool value = (arma::is_Col<armaT>::value || arma::is_Row<armaT>::value);
};

template <typename armaT>
struct is_Arma {
    static constexpr bool value = (arma::is_Mat<armaT>::value || arma::is_Cube<armaT>::value);
};

/* --------------------------------------------------------------
                        Converters
-------------------------------------------------------------- */

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
    static constexpr bool value
        = (is_BorrowConverter<T>::value || is_CopyConverter<T>::value || is_ViewConverter<T>::value
           || is_MoveConverter<T>::value);
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
using iff_Converter = std::enable_if_t<
    is_BorrowConverter<T>::value || is_CopyConverter<T>::value || is_MoveConverter<T>::value
        || is_ViewConverter<T>::value,
    int>;

template <typename T>
using iff_mutable_Converter
    = std::enable_if_t<is_BorrowConverter<T>::value || is_CopyConverter<T>::value || is_MoveConverter<T>::value, int>;

/* --------------------------------------------------------------
                    Memory order policies
-------------------------------------------------------------- */
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
    static constexpr bool value
        = (std::is_same_v<T, CopyResolution> || std::is_same_v<T, RaiseResolution>
           || std::is_same_v<T, CopySwapResolution>);
};

/* --------------------------------------------------------------
                    ConversionConfig
-------------------------------------------------------------- */
template <typename T>
using is_NumpyConversionConfig = internal::is_instance<T, NumpyConversionConfig>;

template <typename T>
using iff_NumpyConversionConfig = std::enable_if_t<is_NumpyConversionConfig<T>::value, int>;

}  // namespace internal
}  // namespace carma
