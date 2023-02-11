#pragma once

#if defined __GNUG__ || defined __clang__  // gnu C++ compiler
#include <cxxabi.h>
#endif  // __GNUG__ || __clang__

#include <armadillo>
#include <carma_bits/type_traits.hpp>
#include <type_traits>
#include <typeinfo>

#if defined(_WIN32) || defined(_WIN64) || defined(WIN32) || defined(__CYGWIN__) || defined(__MINGW32__) || \
    defined(__BORLANDC__)
#define OS_WIN
#endif

// Fix for lack of ssize_t on Windows for >= CPython3.10
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
}  // namespace carma
