include(FetchContent)

SET(DEFAULT_ARMA_VERSION 12.0.x)
IF (NOT USE_ARMA_VERSION)
    MESSAGE(STATUS "carma: Setting Armadillo version to 'v${DEFAULT_ARMA_VERSION}' as none was specified.")
    SET(USE_ARMA_VERSION "${DEFAULT_ARMA_VERSION}" CACHE STRING "Choose the version of Armadillo." FORCE)
    # Set the possible values of build type for cmake-gui
    SET_PROPERTY(CACHE USE_ARMA_VERSION PROPERTY STRINGS
        "10.6.x" "10.8.x" "11.2.x" "11.4.x" "12.2.x"
    )
ENDIF ()

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
