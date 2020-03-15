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
SET(CMAKE_BUILD_TYPE RELEASE)

SET(PROJECT_RELEASEE_FLAGS "-march=native" "-mtune=native" "-mfpmath=sse")
SET(PROJECT_RELEASEE_DEFINITIONS ARMA_NO_DEBUG)
# see linkg for a more expensive approach https://github.com/VcDevel/Vc/blob/master/cmake/OptimizeForArchitecture.cmake

SET(INCLUDE_BASE "${PROJECT_SOURCE_DIR}/tests/include")
SET(ARMA_INCLUDE "${PROJECT_SOURCE_DIR}/tests/include/armadillo/include")
ADD_SUBDIRECTORY("${INCLUDE_BASE}/pybind11")

# ##############################################################################
#                                 DEFINITIONS                                  #
# ##############################################################################
#ADD_DEFINITIONS(-DARMA_DONT_USE_WRAPPER)

# Don't copy arrays if the data is not owned by Numpy
# ADD_DEFINITIONS(-DCARMA_DONT_REQUIRE_OWNDATA)

# Don't copy c-style arrays definition
# ADD_DEFINITIONS(-DCARMA_DONT_REQUIRE_F_CONTIGUOUS)

# ##############################################################################
#                                   INCLUDE                                    #
# ##############################################################################
INCLUDE(${CMAKE_CURRENT_SOURCE_DIR}/tests/CMakeLists.txt)
INCLUDE(${CMAKE_CURRENT_SOURCE_DIR}/examples/CMakeLists.txt)
MESSAGE(STATUS "Build type: " ${CMAKE_BUILD_TYPE})
