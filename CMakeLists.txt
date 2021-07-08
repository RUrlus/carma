CMAKE_MINIMUM_REQUIRED(VERSION 3.16)

SET(CARMA_VERSION_MAJOR 0)
SET(CARMA_VERSION_MINOR 5)
SET(CARMA_VERSION_PATCH 1)
SET(CARMA_VERSION_INFO "${CARMA_VERSION_MAJOR}.${CARMA_VERSION_MINOR}.${CARMA_VERSION_PATCH}")

PROJECT(
    carma
    VERSION ${CARMA_VERSION_INFO}
    LANGUAGES CXX
)

SET(CMAKE_MODULE_PATH_SAVED ${CMAKE_MODULE_PATH})
LIST(INSERT CMAKE_MODULE_PATH 0 "${PROJECT_SOURCE_DIR}/cmake")
# ##############################################################################
#                                   SETTINGS                                   #
# ##############################################################################
SET(CMAKE_CXX_STANDARD 14)
SET(DEFAULT_BUILD_TYPE "Release")
# Set build type to Release if not specified
INCLUDE(BuildType)
INCLUDE(ReleaseSettings)

OPTION(CARMA_DEV_MODE "Enable additional definitions for development" OFF)
OPTION(CARMA_DEV_DEBUG_MODE "Turn on all debugging definitions" OFF)
OPTION(CARMA_BUILD_TESTS "Build tests for carma" OFF)
OPTION(CARMA_BUILD_EXAMPLES "Build examples for carma" OFF)

# collect user settable options
INCLUDE(${PROJECT_SOURCE_DIR}/carma_config.cmake)
INCLUDE(ParseOptions)
CONFIGURE_FILE(
    ${PROJECT_SOURCE_DIR}/include/carma_bits/config.h.in
    ${PROJECT_SOURCE_DIR}/include/carma_bits/config.h
    @ONLY
)
# ##############################################################################
#                                REQUIREMENTS                                  #
# ##############################################################################
FIND_PACKAGE(Python3 COMPONENTS Interpreter Development NumPy REQUIRED)

# Needed to ensure the same Python executable is found by Pybind11
IF (NOT DEFINED PYTHON_EXECUTABLE)
    SET(PYTHON_EXECUTABLE ${Python3_EXECUTABLE})
ENDIF ()
# ##############################################################################
#                                   TARGETS                                    #
# ##############################################################################
# create armadillo and or pybind11 target if not defined
INCLUDE(ThirdPartyTargets)

ADD_LIBRARY(carma INTERFACE)
TARGET_COMPILE_DEFINITIONS(carma
    INTERFACE
    CARMA_VERSION_MAJOR=${CARMA_VERSION_MAJOR}
    CARMA_VERSION_MINOR=${CARMA_VERSION_MINOR}
    CARMA_VERSION_PATCH=${CARMA_VERSION_PATCH}
    CARMA_VERSION_NAME=${CARMA_VERSION_INFO}
    "$<$<CONFIG:RELEASE>:${CARMA_RELEASE_DEFINITIONS}>"
)

TARGET_INCLUDE_DIRECTORIES(carma
    INTERFACE
    # path of the headers after installation
    $<INSTALL_INTERFACE:include> # <prefix>/include/carma
    # path of the headers before installation
    $<BUILD_INTERFACE:${PROJECT_SOURCE_DIR}/include>
)
TARGET_LINK_LIBRARIES(carma
    INTERFACE
        armadillo
        Python3::NumPy
        Python3::Module
        pybind11::pybind11
)
TARGET_PRECOMPILE_HEADERS(carma
    INTERFACE
        ${PROJECT_SOURCE_DIR}/include/carma_bits/cnalloc.hpp
)

SET(CARMA_HEADERS include/carma)

INSTALL(
    DIRECTORY "${PROJECT_SOURCE_DIR}/include" DESTINATION carma
    FILES_MATCHING
        PATTERN "*.h"
        PATTERN "carma"
)

# ##############################################################################
#                              TEST TARGETS                                    #
# ##############################################################################
IF(CARMA_BUILD_TESTS)
    INCLUDE(CTest)
    ENABLE_TESTING()
    INCLUDE(GetCatch2)
    ADD_SUBDIRECTORY(tests)
    INCLUDE(ClangFormatTarget)
    INCLUDE(CoverageTarget)
ENDIF()

IF(CARMA_BUILD_EXAMPLES)
    ADD_SUBDIRECTORY(examples)
ENDIF()
#------------------------------------------------------
INCLUDE(ClearOptions)
SET(CMAKE_MODULE_PATH ${CMAKE_MODULE_PATH_SAVED})
UNSET(CMAKE_MODULE_PATH_SAVED)
