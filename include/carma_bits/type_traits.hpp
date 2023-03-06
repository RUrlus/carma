#pragma once

#include <pybind11/numpy.h>
#include <pybind11/pybind11.h>

#include <armadillo>
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

}  // namespace internal

template <typename numpyT, typename eT>
struct is_Numpy {
    static constexpr bool value = internal::is_same_stripped<numpyT, py::array_t<eT>>::value;
};

template <typename numpyT, typename eT>
using iff_Numpy = std::enable_if_t<is_Numpy<numpyT, eT>::value, int>;

// FIXME
// Use Arma's approach
// template <typename armaT, typename eT>
// struct is_armaMat_only {
//     static constexpr bool value = internal::is_same_stripped<armaT,
//     arma::MatFixed>::value;
// };

template <typename armaT>
using iff_Row = std::enable_if_t<arma::is_Row<armaT>::value, int>;

template <typename armaT>
using iff_Col = std::enable_if_t<arma::is_Col<armaT>::value, int>;

template <typename armaT>
using iff_Vec = std::enable_if_t<arma::is_Col<armaT>::value || arma::is_Row<armaT>::value, int>;

template <typename armaT>
using iff_Mat = std::enable_if_t<arma::is_Mat_only<armaT>::value, int>;

template <typename armaT>
using iff_Cube = std::enable_if_t<arma::is_Cube<armaT>::value, int>;

template <typename armaT>
using iff_Arma = std::enable_if_t<arma::is_Mat<armaT>::value || arma::is_Cube<armaT>::value, int>;

template <typename armaT>
struct is_Vec {
    static constexpr bool value = (arma::is_Col<armaT>::value || arma::is_Row<armaT>::value);
};

template <typename armaT>
struct is_Arma {
    static constexpr bool value = (arma::is_Mat<armaT>::value || arma::is_Cube<armaT>::value);
};

}  // namespace carma
