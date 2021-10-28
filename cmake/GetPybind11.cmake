include(FetchContent)

SET(DEFAULT_PYBIND11_VERSION v2.8.1)
IF (NOT USE_PYBIND11_VERSION)
    MESSAGE(STATUS "carma: Setting Pybind11 version to '${DEFAULT_PYBIND11_VERSION}' as none was specified.")
    SET(USE_PYBIND11_VERSION "${DEFAULT_PYBIND11_VERSION}" CACHE STRING "Choose the version of Pybind11." FORCE)
  # Set the possible values of build type for cmake-gui
  SET_PROPERTY(CACHE USE_PYBIND11_VERSION PROPERTY STRINGS
    "v2.6.0" "v2.6.1" "v2.6.2" "v2.7.1" "v2.8.1"
)
ENDIF ()

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
