include(FetchContent)

SET(DEFAULT_CARMA_VERSION "stable")
IF (NOT USE_CARMA_VERSION)
    MESSAGE(STATUS "carma: Setting carma version to '${DEFAULT_CARMA_VERSION}' as none was specified.")
    SET(USE_CARMA_VERSION "${DEFAULT_CARMA_VERSION}" CACHE STRING "Choose the version of CARMA." FORCE)
    # Set the possible values of build type for cmake-gui
    SET_PROPERTY(CACHE USE_CARMA_VERSION PROPERTY STRINGS
        "stable" "unstable" "0.6.0"  "0.6.1"
    )
ENDIF ()

FetchContent_Declare(
  CarmaCarma
  GIT_REPOSITORY https://github.com/RUrlus/carma.git
  GIT_TAG        ${USE_CARMA_VERSION}
  SOURCE_DIR ${PROJECT_SOURCE_DIR}/extern/carma
)

FetchContent_GetProperties(CarmaCarma)

STRING(TOLOWER "CarmaCarma" lcName)
IF (NOT ${lcName}_POPULATED)
    MESSAGE(STATUS "carma: collecting carma version ${USE_CARMA_VERSION}")
    # Fetch the content using previously declared details
    FetchContent_Populate(CarmaCarma)
ENDIF ()
