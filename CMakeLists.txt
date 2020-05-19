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
SET(DEFAULT_BUILD_TYPE "RELEASE")
IF(NOT CMAKE_BUILD_TYPE AND NOT CMAKE_CONFIGURATION_TYPES)
  MESSAGE(STATUS "Setting build type to '${DEFAULT_BUILD_TYPE}' as none was specified.")
  SET(CMAKE_BUILD_TYPE "${DEFAULT_BUILD_TYPE}" CACHE STRING "Choose the type of build." FORCE)
  # Set the possible values of build type for cmake-gui
  SET_PROPERTY(CACHE CMAKE_BUILD_TYPE PROPERTY STRINGS
    "Debug" "Release" "MinSizeRel" "RelWithDebInfo")
ENDIF()

OPTION(BUILD_TESTS "Build tests for carma" OFF)
OPTION(BUILD_EXAMPLES "Build examples for carma" OFF)
# ##############################################################################
#                                 DEFINITIONS                                  #
# ##############################################################################
#ADD_DEFINITIONS(-DARMA_DONT_USE_WRAPPER)

# Don't copy arrays if the data is not owned by Numpy
# ADD_DEFINITIONS(-DCARMA_DONT_REQUIRE_OWNDATA)

# Don't copy c-style arrays definition
# ADD_DEFINITIONS(-DCARMA_DONT_REQUIRE_F_CONTIGUOUS)

# ##############################################################################
#                                   TARGET                                     #
# ##############################################################################
FIND_PACKAGE(Armadillo)
ADD_LIBRARY(carma INTERFACE)
TARGET_INCLUDE_DIRECTORIES(carma INTERFACE include/ ${ARMADILLO_INCLUDE_DIRS})
TARGET_LINK_LIBRARIES(carma INTERFACE ${ARMADILLO_LIBRARIES})

IF(BUILD_TESTS)
    INCLUDE(${CMAKE_CURRENT_SOURCE_DIR}/tests/CMakeLists.txt)
    MESSAGE(STATUS "Build type: " ${CMAKE_BUILD_TYPE})
ENDIF()
IF(BUILD_EXAMPLES)
    INCLUDE(${CMAKE_CURRENT_SOURCE_DIR}/examples/CMakeLists.txt)
ENDIF()
