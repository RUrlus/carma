cmake_minimum_required(VERSION 3.16)

set(CARMA_VERSION_MAJOR 0)
set(CARMA_VERSION_MINOR 6)
set(CARMA_VERSION_PATCH 7)
set(CARMA_VERSION_INFO
    "${CARMA_VERSION_MAJOR}.${CARMA_VERSION_MINOR}.${CARMA_VERSION_PATCH}")

project(
  carma
  VERSION ${CARMA_VERSION_INFO}
  LANGUAGES CXX)
include(GNUInstallDirs)

set(CMAKE_MODULE_PATH_SAVED ${CMAKE_MODULE_PATH})
list(INSERT CMAKE_MODULE_PATH 0 "${PROJECT_SOURCE_DIR}/cmake")
# ##############################################################################
# SETTINGS                                   #
# ##############################################################################
option(CARMA_INSTALL_LIB "Install CARMA headers" OFF)
set(CARMA_INCLUDE_INSTALL_DIR "${CMAKE_INSTALL_INCLUDEDIR}/carma")

set(DEFAULT_BUILD_TYPE "Release")
# Set build type to Release if not specified
include(BuildType)

if(CMAKE_CXX_COMPILER_ID MATCHES "MSVC")
  # workaround like https://github.com/nlohmann/json/issues/1408 to avoid error
  # like: carma\...\include\armadillo_bits/arma_str.hpp(194): error C2039:
  # '_snprintf': is not a member of 'std'
  add_definitions(-DHAVE_SNPRINTF)
endif()

option(CARMA_DEV_MODE "Enable additional definitions for development" OFF)
option(CARMA_DEV_DEBUG_MODE "Turn on all debugging definitions" OFF)
option(CARMA_BUILD_TESTS "Build tests for carma" OFF)
option(CARMA_BUILD_EXAMPLES "Build examples for carma" OFF)

# collect user settable options
include(${PROJECT_SOURCE_DIR}/carma_config.cmake)
include(ParseOptions)
configure_file(
  ${PROJECT_SOURCE_DIR}/include/carma_bits/config.h.in
  ${PROJECT_BINARY_DIR}/include/carma_bits/generated_config.h @ONLY)

# ##############################################################################
# TARGET                                     #
# ##############################################################################
add_library(carma INTERFACE)
add_library(carma::carma ALIAS carma)

target_compile_definitions(
  carma
  INTERFACE CARMA_USE_GENERATED_CONFIG=1
            CARMA_VERSION_MAJOR=${CARMA_VERSION_MAJOR}
            CARMA_VERSION_MINOR=${CARMA_VERSION_MINOR}
            CARMA_VERSION_PATCH=${CARMA_VERSION_PATCH}
            CARMA_VERSION_NAME=${CARMA_VERSION_INFO}
            "$<$<CONFIG:RELEASE>:${CARMA_RELEASE_DEFINITIONS}>")
target_compile_features(carma INTERFACE cxx_std_14)

if(CARMA_DEV_MODE)
  target_compile_options(carma INTERFACE -Wall -Wextra -Wunused-variable
                                         -Wunused-const-variable)
endif()

target_include_directories(
  carma
  INTERFACE $<INSTALL_INTERFACE:${CARMA_INCLUDE_INSTALL_DIR}>
            $<BUILD_INTERFACE:${PROJECT_SOURCE_DIR}/include>
            $<BUILD_INTERFACE:${PROJECT_BINARY_DIR}/include>)

target_link_libraries(carma INTERFACE armadillo::armadillo Python3::NumPy
                                      Python3::Module pybind11::pybind11)

add_library(carma_headers INTERFACE)
add_library(carma::headers ALIAS carma_headers)

target_compile_definitions(
  carma_headers
  INTERFACE CARMA_VERSION_MAJOR=${CARMA_VERSION_MAJOR}
            CARMA_VERSION_MINOR=${CARMA_VERSION_MINOR}
            CARMA_VERSION_PATCH=${CARMA_VERSION_PATCH}
            CARMA_VERSION_NAME=${CARMA_VERSION_INFO})
target_compile_features(carma_headers INTERFACE cxx_std_14)

target_include_directories(
  carma_headers
  INTERFACE $<INSTALL_INTERFACE:${CARMA_INCLUDE_INSTALL_DIR}>
            $<BUILD_INTERFACE:${PROJECT_SOURCE_DIR}/include>
            $<BUILD_INTERFACE:${PROJECT_BINARY_DIR}/include>)

# Only install the headers
if(CARMA_INSTALL_LIB)
  # modified from https://dominikberner.ch/cmake-interface-lib/

  # locations are provided by GNUInstallDirs
  install(
    TARGETS carma
    EXPORT carma_Targets
    ARCHIVE DESTINATION ${CMAKE_INSTALL_LIBDIR}
    LIBRARY DESTINATION ${CMAKE_INSTALL_LIBDIR}
    RUNTIME DESTINATION ${CMAKE_INSTALL_BINDIR})

  install(
    TARGETS carma_headers
    EXPORT carma_Header_Targets
    ARCHIVE DESTINATION ${CMAKE_INSTALL_LIBDIR}
    LIBRARY DESTINATION ${CMAKE_INSTALL_LIBDIR}
    RUNTIME DESTINATION ${CMAKE_INSTALL_BINDIR})

  include(CMakePackageConfigHelpers)
  write_basic_package_version_file(
    "carmaConfigVersion.cmake"
    VERSION ${PROJECT_VERSION}
    COMPATIBILITY SameMinorVersion)

  configure_package_config_file(
    "${PROJECT_SOURCE_DIR}/cmake/carmaConfig.cmake.in"
    "${PROJECT_BINARY_DIR}/carmaConfig.cmake"
    INSTALL_DESTINATION ${CMAKE_INSTALL_DATAROOTDIR}/carma/cmake)

  install(
    EXPORT carma_Targets
    FILE carmaCarmaTargets.cmake
    NAMESPACE carma::
    DESTINATION ${CMAKE_INSTALL_DATAROOTDIR}/carma/cmake)

  install(
    EXPORT carma_Header_Targets
    FILE carmaHeaderTargets.cmake
    NAMESPACE carma::
    DESTINATION ${CMAKE_INSTALL_DATAROOTDIR}/carma/cmake)

  install(
    FILES "${PROJECT_BINARY_DIR}/carmaConfig.cmake"
          "${PROJECT_BINARY_DIR}/carmaConfigVersion.cmake"
          "${PROJECT_SOURCE_DIR}/cmake/carmaDependencyTargets.cmake"
    DESTINATION ${CMAKE_INSTALL_DATAROOTDIR}/carma/cmake)

  install(FILES "${PROJECT_SOURCE_DIR}/include/carma"
          DESTINATION ${CARMA_INCLUDE_INSTALL_DIR})

  install(
    DIRECTORY ${PROJECT_SOURCE_DIR}/include/carma_bits
    DESTINATION ${CARMA_INCLUDE_INSTALL_DIR}
    FILES_MATCHING
    PATTERN "*.h")

  install(FILES ${PROJECT_BINARY_DIR}/include/carma_bits/generated_config.h
          DESTINATION ${CARMA_INCLUDE_INSTALL_DIR}/carma_bits)

else()
  # ############################################################################
  # REQUIREMENTS                                  #
  # ############################################################################
  find_package(
    Python3
    COMPONENTS Development NumPy
    QUIET)
  if(NOT Python3_FOUND)
    find_package(
      Python3
      COMPONENTS Development.Module NumPy
      REQUIRED)
  endif()

  # Needed to ensure the same Python executable is found by Pybind11
  if(NOT DEFINED PYTHON_EXECUTABLE)
    set(PYTHON_EXECUTABLE ${Python3_EXECUTABLE})
  endif()

  # create armadillo and or pybind11 target if not defined
  include(ThirdPartyTargets)

  target_precompile_headers(carma INTERFACE
                            ${PROJECT_SOURCE_DIR}/include/carma_bits/cnalloc.h)
endif()
# ##############################################################################
# TEST TARGETS                                    #
# ##############################################################################

if(CARMA_BUILD_TESTS OR CARMA_BUILD_EXAMPLES)
  set(CMAKE_CXX_STANDARD 14)
endif()

if(CARMA_BUILD_TESTS)
  include(CTest)
  enable_testing()
  include(GetCatch2)
  add_subdirectory(tests)
  include(ClangFormatTarget)
  include(CoverageTarget)
endif()

if(CARMA_BUILD_EXAMPLES)
  add_subdirectory(examples)
endif()
# ------------------------------------------------------
include(ClearOptions)
set(CMAKE_MODULE_PATH ${CMAKE_MODULE_PATH_SAVED})
unset(CMAKE_MODULE_PATH_SAVED)
