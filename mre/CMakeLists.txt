CMAKE_MINIMUM_REQUIRED(VERSION 3.16 FATAL_ERROR)

PROJECT(carma_bug_mre LANGUAGES CXX)
LIST(INSERT CMAKE_MODULE_PATH 0 "${PROJECT_SOURCE_DIR}/../cmake")

# creates the following targets needed by CARMA
# pybind11::pybind11
# armadillo::armadillo
INCLUDE(ThirdPartyTargets)

# create carma target
INCLUDE(GetCarma)
ADD_SUBDIRECTORY("${PROJECT_SOURCE_DIR}/extern/carma")

# create target python module with Pybind11 
pybind11_add_module(pymod MODULE mre.cpp)
TARGET_INCLUDE_DIRECTORIES(pymod PUBLIC mre.h)
TARGET_LINK_LIBRARIES(pymod PUBLIC carma::carma)
TARGET_COMPILE_DEFINITIONS(pymod PUBLIC ARMA_EXTRA_DEBUG CARMA_EXTRA_DEBUG CARMA_DEV_DEBUG)

FILE(TOUCH "${PROJECT_BINARY_DIR}/__init__.py")

INSTALL(TARGETS pymod DESTINATION ${PROJECT_BINARY_DIR})
