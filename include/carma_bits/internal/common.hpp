#pragma once

#if defined __GNUG__ || defined __clang__  // gnu C++ compiler
#include <cxxabi.h>
#endif  // __GNUG__ || __clang__

#include <memory>
#include <string>
#include <type_traits>
#include <typeinfo>

#ifdef CARMA_DEBUG
#include <iostream>
#endif  // CARMA_DEBUG

#if defined(_WIN32) || defined(_WIN64) || defined(WIN32) || defined(__CYGWIN__) || defined(__MINGW32__) \
    || defined(__BORLANDC__)
#define OS_WIN
#endif

#if defined(_MSC_VER)
#pragma warning(push)
#pragma warning(disable : 4127)  // warning C4127: Conditional expression is constant
// FIXME check if we need this...
// Fix for lack of ssize_t on Windows for >= CPython3.10
// #include <BaseTsd.h>
// typedef SSIZE_T ssize_t;
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

namespace carma::internal {

#if defined(__GNUG__) || defined(__clang__)

inline std::string demangle(const char* mangled_name) {
    std::size_t len = 0;
    int status = 0;
    std::unique_ptr<char, decltype(&std::free)> ptr(
        __cxxabiv1::__cxa_demangle(mangled_name, nullptr, &len, &status), &std::free
    );
    return ptr.get();
}

#else

inline std::string demangle(const char* name) { return name; }

#endif  // __GNUG__ || __clang__

template <typename T>
inline std::string get_full_typename() {
    std::string name;
    if (std::is_const_v<std::remove_reference_t<T>>)
        name += "const ";
    name += demangle(typeid(T).name());
    if (std::is_lvalue_reference_v<T>) {
        name += "&";
    } else if (std::is_rvalue_reference_v<T>) {
        name += "&&";
    }
    return name;
}

#ifndef CARMA_DEBUG
template <typename... Args>
inline void carma_debug_print(Args...) {}
#else
template <typename... Args>
inline void carma_debug_print(Args... args) {
    std::cout << "|carma| ";
    (std::cout << ... << args) << "\n";
}
#endif

#ifndef CARMA_EXTRA_DEBUG
template <typename... Args>
inline void carma_extra_debug_print(Args...) {}
#else
template <typename... Args>
inline void carma_extra_debug_print(Args... args) {
    std::cout << "|carma| ";
    (std::cout << ... << args) << "\n";
}
#endif

}  // namespace carma::internal
