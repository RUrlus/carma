CMAKE_MINIMUM_REQUIRED(VERSION 3.14.4)

PROJECT(
    carma
    VERSION 0.1.0
    DESCRIPTION "Header only library for conversion between numpy arrays and armadillo matrices based on pybind11"
    LANGUAGES CXX
)

# ##############################################################################
#                                   SETTINGS                                   #
# ##############################################################################
SET(CMAKE_CXX_STANDARD 14)
SET(CMAKE_BUILD_TYPE DEBUG)
OPTION(BUILD_TESTS "Build tests for carma" ON)

SET(INCLUDE_BASE "${CMAKE_CURRENT_SOURCE_DIR}/include")
SET(ARMA_INCLUDE "${CMAKE_CURRENT_SOURCE_DIR}/include/armadillo/include")

SET(PROJECT_RELEASEE_FLAGS "-march=native" "-mtune=native" "-mfpmath=sse")
SET(PROJECT_RELEASEE_DEFINITIONS ARMA_NO_DEBUG)
# see linkg for a more expensive approach https://github.com/VcDevel/Vc/blob/master/cmake/OptimizeForArchitecture.cmake

# ##############################################################################
#                                 DEFINITIONS                                  #
# ##############################################################################
ADD_DEFINITIONS(-DARMA_DONT_USE_WRAPPER)

# ##############################################################################
#                                   INCLUDE                                    #
# ##############################################################################
ADD_SUBDIRECTORY("${INCLUDE_BASE}/pybind11")
INCLUDE(${CMAKE_CURRENT_SOURCE_DIR}/tests/CMakeLists.txt)
MESSAGE(STATUS "Build type: " ${CMAKE_BUILD_TYPE})
