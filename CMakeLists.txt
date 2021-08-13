CMAKE_MINIMUM_REQUIRED(VERSION 3.16)

SET(CARMA_VERSION_MAJOR 0)
SET(CARMA_VERSION_MINOR 5)
SET(CARMA_VERSION_PATCH 2)
SET(CARMA_VERSION_INFO "${CARMA_VERSION_MAJOR}.${CARMA_VERSION_MINOR}.${CARMA_VERSION_PATCH}")

PROJECT(
    carma
    VERSION ${CARMA_VERSION_INFO}
    LANGUAGES CXX
)
INCLUDE(GNUInstallDirs)

SET(CMAKE_MODULE_PATH_SAVED ${CMAKE_MODULE_PATH})
LIST(INSERT CMAKE_MODULE_PATH 0 "${PROJECT_SOURCE_DIR}/cmake")
# ##############################################################################
#                                   SETTINGS                                   #
# ##############################################################################
OPTION(CARMA_INSTALL_LIB "Install CARMA headers" OFF)
SET(CARMA_INCLUDE_INSTALL_DIR "${CMAKE_INSTALL_INCLUDEDIR}/carma")

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
#                                   TARGET                                     #
# ##############################################################################
ADD_LIBRARY(carma INTERFACE)
ADD_LIBRARY(carma::carma ALIAS carma)

TARGET_COMPILE_DEFINITIONS(carma
    INTERFACE
    CARMA_VERSION_MAJOR=${CARMA_VERSION_MAJOR}
    CARMA_VERSION_MINOR=${CARMA_VERSION_MINOR}
    CARMA_VERSION_PATCH=${CARMA_VERSION_PATCH}
    CARMA_VERSION_NAME=${CARMA_VERSION_INFO}
    "$<$<CONFIG:RELEASE>:${CARMA_RELEASE_DEFINITIONS}>"
)
IF(CARMA_DEV_MODE)
    TARGET_COMPILE_OPTIONS(carma
        INTERFACE
        -Wall
        -Wextra
        -Wunused-variable
        -Wunused-const-variable
    )
ENDIF()

TARGET_INCLUDE_DIRECTORIES(carma
    INTERFACE
    $<INSTALL_INTERFACE:${CARMA_INCLUDE_INSTALL_DIR}>
    $<BUILD_INTERFACE:${PROJECT_SOURCE_DIR}/include>
)

TARGET_LINK_LIBRARIES(carma
    INTERFACE
        armadillo::armadillo
        Python3::NumPy
        Python3::Module
        pybind11::pybind11
)

# Only install the headers
IF (CARMA_INSTALL_LIB)
    # modified from https://dominikberner.ch/cmake-interface-lib/

    # locations are provided by GNUInstallDirs
    INSTALL(
        TARGETS carma
        EXPORT carma_Targets
        ARCHIVE DESTINATION ${CMAKE_INSTALL_LIBDIR}
        LIBRARY DESTINATION ${CMAKE_INSTALL_LIBDIR}
        RUNTIME DESTINATION ${CMAKE_INSTALL_BINDIR}
    )
    
    INCLUDE(CMakePackageConfigHelpers)
    write_basic_package_version_file(
        "carmaConfigVersion.cmake"
        VERSION ${PROJECT_VERSION}
        COMPATIBILITY SameMinorVersion
    )
    
    configure_package_config_file(
        "${PROJECT_SOURCE_DIR}/cmake/carmaConfig.cmake.in"
        "${PROJECT_BINARY_DIR}/carmaConfig.cmake"
        INSTALL_DESTINATION ${CMAKE_INSTALL_DATAROOTDIR}/carma/cmake
    )
    
    INSTALL(
      EXPORT carma_Targets
      FILE carmaTargets.cmake
      NAMESPACE carma::
      DESTINATION ${CMAKE_INSTALL_DATAROOTDIR}/carma/cmake
    )
    
    INSTALL(
        FILES
            "${PROJECT_BINARY_DIR}/carmaConfig.cmake"
            "${PROJECT_BINARY_DIR}/carmaConfigVersion.cmake"
        DESTINATION ${CMAKE_INSTALL_DATAROOTDIR}/carma/cmake
    )

    INSTALL(
        FILES "${PROJECT_SOURCE_DIR}/include/carma"
        DESTINATION ${CARMA_INCLUDE_INSTALL_DIR}
    )

    INSTALL(
        DIRECTORY ${PROJECT_SOURCE_DIR}/include/carma_bits
        DESTINATION ${CARMA_INCLUDE_INSTALL_DIR}
        FILES_MATCHING
            PATTERN "*.h"
    )
    
#INSTALL(DIRECTORY ${PROJECT_BINARY_DIR}/tmp/ DESTINATION ${CARMA_INCLUDE_INSTALL_DIR})
ELSE()
    # create armadillo and or pybind11 target if not defined
    INCLUDE(ThirdPartyTargets)

    TARGET_PRECOMPILE_HEADERS(carma
        INTERFACE
            ${PROJECT_SOURCE_DIR}/include/carma_bits/cnalloc.h
    )
ENDIF()
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
