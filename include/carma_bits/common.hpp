#ifndef INCLUDE_CARMA_BITS_COMMON_HPP_
#define INCLUDE_CARMA_BITS_COMMON_HPP_

#if defined __GNUG__ || defined __clang__  // gnu C++ compiler
#include <cxxabi.h>
#endif  // __GNUG__ || __clang__

#include <pybind11/numpy.h>
#include <pybind11/pybind11.h>

#include <armadillo>
#include <type_traits>
#include <typeinfo>

#if defined(_WIN32) || defined(_WIN64) || defined(WIN32) || defined(__CYGWIN__) || defined(__MINGW32__) || \
    defined(__BORLANDC__)
#define OS_WIN
#endif

// Fix for lack of ssize_t on Windows for CPython3.10
#if defined(_MSC_VER)
#pragma warning(push)
#pragma warning(disable : 4127)  // warning C4127: Conditional expression is constant
#include <BaseTsd.h>
typedef SSIZE_T ssize_t;
#endif

#ifndef CARMA_DEFINED_EXPECT
#if defined(__has_builtin) && __has_builtin(__builtin_expect)
#define CARMA_UNLIKELY(expr) __builtin_expect(!!(expr), 0)
#define CARMA_LIKELY(expr) __builtin_expect(!!(expr), 1)
#define CARMA_DEFINED_EXPECT
#elif defined(__GNUC__) && (__GNUC__ > 3 || __GNUC__ == 3)
#define CARMA_UNLIKELY(expr) __builtin_expect(!!(expr), 0)
#define CARMA_LIKELY(expr) __builtin_expect(!!(expr), 1)
#define CARMA_DEFINED_EXPECT
#else
#define CARMA_UNLIKELY(expr) (!!(expr))
#define CARMA_LIKELY(expr) (!!(expr))
#define CARMA_DEFINED_EXPECT
#endif  // CARMA_LIKELY
#endif  // CARMA_DEFINED_EXPECT

namespace carma {

namespace py = pybind11;

namespace internal {
#if defined(__GNUG__) || defined(__clang__)

inline std::string demangle(const char* mangled_name) {
    std::size_t len = 0;
    int status = 0;
    std::unique_ptr<char, decltype(&std::free)> ptr(__cxxabiv1::__cxa_demangle(mangled_name, nullptr, &len, &status),
                                                    &std::free);
    return ptr.get();
}

#else

inline std::string demangle(const char* name) { return name; }

#endif  // __GNUG__ || __clang__

template <typename T>
inline std::string get_full_typename() {
    std::string name;
    if (std::is_const_v<std::remove_reference_t<T>>) name += "const ";
    name += demangle(typeid(T).name());
    if (std::is_lvalue_reference_v<T>) {
        name += "&";
    } else if (std::is_rvalue_reference_v<T>) {
        name += "&&";
    }
    return name;
}

}  // namespace internal

/* -----------------------------------------------------------------------------
                                   Type traits
----------------------------------------------------------------------------- */
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

template <typename T>
using iff_const = std::enable_if_t<std::is_const_v<T>, int>;

}  // namespace internal

template <typename numpyT, typename eT>
using iff_Numpy = std::enable_if_t<std::is_same_v<std::remove_cv_t<std::remove_reference_t<numpyT>>, py::array_t<eT>> ||
                                       std::is_same_v<std::remove_cv_t<numpyT>, py::array_t<eT>*>,
                                   int>;

template <typename numpyT, typename eT>
struct is_Numpy {
    static constexpr bool value = (std::is_same_v<std::remove_cv_t<std::remove_reference_t<numpyT>>, py::array_t<eT>> ||
                                   std::is_same_v<std::remove_cv_t<numpyT>, py::array_t<eT>*>);
};

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
struct is_Arma {
    static constexpr bool value = (arma::is_Mat<armaT>::value || arma::is_Cube<armaT>::value);
};

}  // namespace carma
#endif  // INCLUDE_CARMA_BITS_NPTOARMA_HPP_
