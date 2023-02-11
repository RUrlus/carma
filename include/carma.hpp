/*  carma/carma: Bidirectional coverter of Numpy arrays and Armadillo objects
 *  Copyright (c) 2022 Ralph Urlus <rurlus.dev@gmail.com>
 *  All rights reserved. Use of this source code is governed by a
 *  Apache-2.0 license that can be found in the LICENSE file.
 */
#ifndef INCLUDE_CARMA_
#define INCLUDE_CARMA_

/* If the Numpy allocator/deallocator have not been set through
 * the carma_armadillo target ARMA_ALIEN_MEM_ALLOC_FUNCTION and
 * ARMA_ALIEN_MEM_FREE_FUNCTION need to be set.
 *
 * This requires that Armadillo wasn't included before carma
 * The CMake script handles this by pre-compiling the numpy_alloc header
 */
#ifndef CARMA_ARMA_ALIEN_MEM_FUNCTIONS_SET
#if defined(ARMA_VERSION_MAJOR)
#error "|carma| please include the armadillo header after the carma header or use carma's CMake build"
#endif
#include <carma_bits/numpy_alloc.hpp>
#endif

#include <armadillo>
#ifdef CARMA_EXTRA_DEBUG
#include <iostream>
#endif  // CARMA_EXTRA_DEBUG

#include <carma_bits/converters.hpp>

#if defined(CARMA_ARMA_ALIEN_MEM_FUNCTIONS_SET)
#if ((!defined(ARMA_ALIEN_MEM_ALLOC_FUNCTION)) || (!defined(ARMA_ALIEN_MEM_FREE_FUNCTION)))
#error \
    "|carma| ARMA_ALIEN_MEM_ALLOC_FUNCTION and or ARMA_ALIEN_MEM_FREE_FUNCTION not set while CARMA_ARMA_ALIEN_MEM_FUNCTIONS_SET is"
#endif
#endif

#define CARMA_ARMA_VERSION (ARMA_VERSION_MAJOR * 10000 + ARMA_VERSION_MINOR * 100 + ARMA_VERSION_PATCH)
static_assert(CARMA_ARMA_VERSION > 100502, "|carma| minimum supported armadillo version is 10.5.2");

#ifndef CARMA_VERSION_MAJOR
#define CARMA_VERSION_MAJOR 0
#define CARMA_VERSION_MINOR 7
#define CARMA_VERSION_PATCH 0
#define CARMA_VERSION_NAME "0.7.0 HO"
#endif

namespace carma {

struct carma_version {
    static constexpr unsigned int major = CARMA_VERSION_MAJOR;
    static constexpr unsigned int minor = CARMA_VERSION_MINOR;
    static constexpr unsigned int patch = CARMA_VERSION_PATCH;

    static inline std::string as_string() {
        std::ostringstream buffer;
        buffer << carma_version::major << "." << carma_version::minor << "." << carma_version::patch;
        return buffer.str();
    }
};  // carma_version

#if defined(CARMA_EXTRA_DEBUG)

#ifndef CARMA_DEBUG
#define CARMA_DEBUG true
#endif  // CARMA_DEBUG

namespace anon {
class carma_config_debug_message {
   public:
    inline carma_config_debug_message() {
        std::cout << "\n|----------------------------------------------------------|\n"
                  << "|                    CARMA CONFIGURATION                   |"
                  << "\n|----------------------------------------------------------|\n|\n";
        std::cout << "| Carma version: " + carma_version().as_string() << "\n|\n";
        std::cout << "| Default Numpy to Arma conversion config:\n"
                  << "| ----------------------------------------\n"
                  << "| * l-value converter:                 " << CARMA_DEFAULT_LVALUE_CONVERTER::name_ << "\n"
                  << "| * const l-value converter:           " << CARMA_DEFAULT_CONST_LVALUE_CONVERTER::name_ << "\n"
                  << "| * resolution_policy:                 " << CARMA_DEFAULT_RESOLUTION::name_ << "\n"
                  << "| * memory_order_policy:               " << CARMA_DEFAULT_MEMORY_ORDER::name_ << "\n";
        std::cout << "|\n| Converter Options:\n"
                  << "| ------------------\n"
                  << "| * enforce rvalue for MoveConverter:  "
#ifndef CARMA_DONT_ENFORCE_RVALUE_MOVECONVERTER
                  << "true\n";
#else
                  << "false\n";
#endif  // CARMA_DONT_ENFORCE_RVALUE_MOVECONVERTER
        std::cout << "|\n|----------------------------------------------------------|\n\n";
    };
};

static carma_config_debug_message carma_config_debug_message_print;
}  // namespace anon

#endif  // CARMA_EXTRA_DEBUG

}  // namespace carma

#endif  // INCLUDE_CARMA_
