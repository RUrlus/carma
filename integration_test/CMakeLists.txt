CMAKE_MINIMUM_REQUIRED(VERSION 3.16)

PROJECT(
    integration_test_carma
    VERSION 0.1.0
    LANGUAGES CXX
)

SET(MODNAME "integration_test_carma")

# -- EXTERNAL
INCLUDE(FetchContent)

SET(USE_PYBIND11_VERSION "v2.6.2")

# Pybind11
FetchContent_Declare(
  Pybind11Repo
  GIT_REPOSITORY https://github.com/pybind/pybind11.git
  GIT_TAG        ${USE_PYBIND11_VERSION}
  SOURCE_DIR ${PROJECT_SOURCE_DIR}/extern/pybind11
)

FetchContent_GetProperties(Pybind11Repo)

STRING(TOLOWER "Pybind11Repo" lcName)
IF (NOT ${lcName}_POPULATED)
  MESSAGE(STATUS "carma: collecting Pybind11 ${USE_PYBIND11_VERSION}")
  # Fetch the content using previously declared details
  FetchContent_Populate(Pybind11Repo)
ENDIF ()

ADD_SUBDIRECTORY(extern/pybind11)

SET(USE_ARMA_VERSION 10.6.x)
FetchContent_Declare(
  CarmaArmadillo
  GIT_REPOSITORY https://gitlab.com/conradsnicta/armadillo-code.git
  GIT_TAG        ${USE_ARMA_VERSION}
  SOURCE_DIR ${PROJECT_SOURCE_DIR}/extern/armadillo-code
)

FetchContent_GetProperties(CarmaArmadillo)

STRING(TOLOWER "CarmaArmadillo" lcName)
IF (NOT ${lcName}_POPULATED)
    MESSAGE(STATUS "carma: collecting Armadillo ${USE_ARMA_VERSION}")
    # Fetch the content using previously declared details
    FetchContent_Populate(CarmaArmadillo)
ENDIF ()

SET(ARMADILLO_INCLUDE_DIR extern/armadillo-code/include)

IF (CARMA_TEST_MODE STREQUAL "comp_carma")
    FIND_PACKAGE(carma CONFIG REQUIRED COMPONENTS carma)
ELSEIF (CARMA_TEST_MODE STREQUAL "comp_headers")
    FIND_PACKAGE(carma CONFIG REQUIRED COMPONENTS headers)
ELSE ()
    FIND_PACKAGE(carma CONFIG REQUIRED)
ENDIF ()

INCLUDE(CTest)
ENABLE_TESTING()

# -- TARGET
pybind11_add_module(${MODNAME}
    MODULE
        src/integration_test.cpp
)

IF (CARMA_TEST_MODE STREQUAL "comp_headers")
    TARGET_LINK_LIBRARIES(${MODNAME}
        PRIVATE
            carma::headers
            armadillo::armadillo
            Python3::NumPy
            Python3::Module
            pybind11::pybind11
    )
ELSE ()
    TARGET_LINK_LIBRARIES(${MODNAME} PUBLIC carma::carma)
ENDIF ()
TARGET_INCLUDE_DIRECTORIES(${MODNAME}
    PUBLIC
        $<BUILD_INTERFACE:${PROJECT_SOURCE_DIR}/src>
)

# -- TEST
INSTALL(TARGETS ${MODNAME} DESTINATION tests)
FILE(GLOB PY_TEST_FILES "${PROJECT_SOURCE_DIR}/tests/*.py")
INSTALL(FILES ${PY_TEST_FILES} DESTINATION tests)

ADD_TEST(NAME pytest
         WORKING_DIRECTORY ${CMAKE_CURRENT_SOURCE_DIR}/tests
         COMMAND ${Python3_EXECUTABLE} -m pytest -vv
)

SET_PROPERTY(TEST pytest
    APPEND PROPERTY ENVIRONMENT "PYTHONPATH=${CMAKE_CURRENT_BINARY_DIR}/tests"
)
