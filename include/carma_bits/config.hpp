#pragma once

#ifdef CARMA_EXTRA_DEBUG
#ifndef CARMA_DEBUG
#define CARMA_DEBUG true
#endif  // CARMA_DEBUG
#endif  // CARMA_EXTRA_DEBUG

#ifdef CARMA_EXTENSION_MODE
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
#include <carma_bits/extension/numpy_alloc.hpp>
#endif  //  CARMA_ARMA_ALIEN_MEM_FUNCTIONS_SET

#ifdef CARMA_ARMA_ALIEN_MEM_FUNCTIONS_SET
#if ((!defined(ARMA_ALIEN_MEM_ALLOC_FUNCTION)) || (!defined(ARMA_ALIEN_MEM_FREE_FUNCTION)))
#error \
    "|carma| ARMA_ALIEN_MEM_ALLOC_FUNCTION and or ARMA_ALIEN_MEM_FREE_FUNCTION not set while CARMA_ARMA_ALIEN_MEM_FUNCTIONS_SET is"
#endif
#endif  // CARMA_ARMA_ALIEN_MEM_FUNCTIONS_SET

#include <carma_bits/extension/config.hpp>
#else
#include <carma_bits/base/config.hpp>
#endif  // CARMA_EXTENSION_MODE

#include <armadillo>
#define CARMA_ARMA_VERSION (ARMA_VERSION_MAJOR * 10000 + ARMA_VERSION_MINOR * 100 + ARMA_VERSION_PATCH)
static_assert(CARMA_ARMA_VERSION > 100502, "|carma| minimum supported armadillo version is 10.5.2");

#include <carma_bits/version.hpp>
