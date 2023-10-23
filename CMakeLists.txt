cmake_minimum_required(VERSION 3.20)

set(CARMA_VERSION_MAJOR 1)
set(CARMA_VERSION_MINOR 0)
set(CARMA_VERSION_PATCH 0)
set(CARMA_VERSION_INFO
    "${CARMA_VERSION_MAJOR}.${CARMA_VERSION_MINOR}.${CARMA_VERSION_PATCH}")

project(
  carma
  VERSION ${CARMA_VERSION_INFO}
  LANGUAGES CXX)
include(GNUInstallDirs)

# ##############################################################################
# SETTINGS                                   #
# ##############################################################################
# add_subdirectory(${PROJECT_SOURCE_DIR}/extern/armadillo-code)
set(CMAKE_CXX_STANDARD
    17
    CACHE STRING "C++ version selection")
set(CMAKE_CXX_STANDARD_REQUIRED ON) # optional, ensure standard is supported
set(CMAKE_CXX_EXTENSIONS OFF) # optional, keep compiler extensions off

if(NOT DEFINED CMAKE_BUILD_TYPE)
  set(CMAKE_BUILD_TYPE "Release")
endif()

# find_program(CCACHE_FOUND ccache) if(CCACHE_FOUND) set_property(GLOBAL
# PROPERTY RULE_LAUNCH_COMPILE ccache) endif(CCACHE_FOUND)

# if (CMAKE_CXX_COMPILER_ID MATCHES "MSVC") # workaround like
# https://github.com/nlohmann/json/issues/1408 # to avoid error like:
# carma\...\include\armadillo_bits/arma_str.hpp(194): # error C2039:
# '_snprintf': is not a member of 'std' add_definitions(-DHAVE_SNPRINTF) endif
# ()

# Create the public carma header
# configure_file("${PROJECT_SOURCE_DIR}/include/carma.hpp"
# "${PROJECT_SOURCE_DIR}/include/carma" COPYONLY)

# ##############################################################################
# REQUIREMENTS                                  #
# ##############################################################################

find_package(
  Python3
  COMPONENTS Interpreter Development NumPy
  REQUIRED)
if(NOT DEFINED PYTHON_EXECUTABLE)
  set(PYTHON_EXECUTABLE ${Python3_EXECUTABLE})
endif()
find_package(pybind11 REQUIRED)
find_package(Armadillo REQUIRED)

add_library(carma INTERFACE)
add_library(carma_expansion INTERFACE)
add_library(carma_headers INTERFACE)
add_library(carma::carma ALIAS carma)
add_library(carma::headers ALIAS carma_headers)
add_library(carma::expansion ALIAS carma_expansion)

target_compile_definitions(
  carma
  INTERFACE CARMA_VERSION_MAJOR=${CARMA_VERSION_MAJOR}
            CARMA_VERSION_MINOR=${CARMA_VERSION_MINOR}
            CARMA_VERSION_PATCH=${CARMA_VERSION_PATCH}
            CARMA_VERSION_NAME=${CARMA_VERSION_INFO})
set_target_properties(carma PROPERTIES CXX_STANDARD_REQUIRED ON)

target_compile_options(carma INTERFACE -Wall -Wextra -Wunused-variable
                                       -Wunused-const-variable)
if(CARMA_ENABLE_EXTRA_DEBUG)
  target_compile_definitions(carma INTERFACE CARMA_EXTRA_DEBUG=true
                                             CARMA_DEBUG=true)
  set(ignoreMe "${CARMA_ENABLE_DEBUG}")
elseif(CARMA_ENABLE_DEBUG)
  target_compile_definitions(carma INTERFACE CARMA_DEBUG=true)
endif()

target_include_directories(
  carma INTERFACE $<BUILD_INTERFACE:${PROJECT_SOURCE_DIR}/include>
                  ${ARMADILLO_INCLUDE_DIRS})

target_link_libraries(carma INTERFACE ${ARMADILLO_LIBRARIES} pybind11::pybind11
                                      Python3::NumPy Python3::Module)

add_executable(main ${PROJECT_SOURCE_DIR}/src/main.cpp)

target_link_libraries(main carma pybind11::embed)

install(TARGETS main DESTINATION "${PROJECT_SOURCE_DIR}/bin")
