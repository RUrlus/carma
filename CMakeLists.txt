CMAKE_MINIMUM_REQUIRED(VERSION 3.14)

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
if (CMAKE_CXX_COMPILER_ID STREQUAL "GNU" 
        OR CMAKE_CXX_COMPILER_ID STREQUAL "Clang" 
        OR CMAKE_CXX_COMPILER_ID STREQUAL "AppleClang")
    SET(PROJECT_RELEASE_FLAGS "-march=native" "-mtune=native")
elseif (CMAKE_CXX_COMPILER_ID MATCHES "MSVC")
    # workaround like https://github.com/nlohmann/json/issues/1408
    # to avoid error like: carma\third_party\armadillo-code\include\armadillo_bits/arma_str.hpp(194): error C2039: '_snprintf': is not a member of 'std' (compiling source file carma\tests\src\bindings.cpp) 
    ADD_DEFINITIONS(-DHAVE_SNPRINTF)
endif ()
SET(PROJECT_RELEASE_DEFINITIONS ARMA_NO_DEBUG)
# see linkg for a more expensive approach https://github.com/VcDevel/Vc/blob/master/cmake/OptimizeForArchitecture.cmake


ADD_LIBRARY(carma INTERFACE)
TARGET_LINK_LIBRARIES(carma INTERFACE pybind11 armadillo)
TARGET_INCLUDE_DIRECTORIES(carma
        INTERFACE
        # path of the headers after installation
        $<INSTALL_INTERFACE:include> # <prefix>/include/carma
        # path of the headers before installation
        $<BUILD_INTERFACE:${PROJECT_SOURCE_DIR}/include>
        )
INSTALL(DIRECTORY "${PROJECT_SOURCE_DIR}/include" 
        DESTINATION .
        FILES_MATCHING PATTERN "*.h")


IF (BUILD_TESTS OR BUILD_EXAMPLES)
    MESSAGE(STATUS "Build type: " ${CMAKE_BUILD_TYPE})

    # ##############################################################################
    #                             THIRD PARTY LIBS                                 #
    # ##############################################################################
    # Use a header only version of armadillo; default CMake project is not header only
    # That's why we use a custom 'armadillo' target with only include directory
    #add_subdirectory(third_party/armadillo-code)
    SET(ARMADILLO_INCLUDE_DIR third_party/armadillo-code/include)
    ADD_LIBRARY(armadillo INTERFACE)
    target_include_directories(armadillo INTERFACE ${ARMADILLO_INCLUDE_DIR})
    
    add_subdirectory(third_party/pybind11)


    ENABLE_TESTING()
    FIND_PACKAGE(Python3)
ENDIF()

IF(BUILD_TESTS)
    add_subdirectory(tests)
ENDIF()
IF(BUILD_EXAMPLES)
    add_subdirectory(examples)
ENDIF()
