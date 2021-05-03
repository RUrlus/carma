include(FetchContent)

FetchContent_Declare(
  ArmadilloFork
  GIT_REPOSITORY https://gitlab.com/RUrlus/armadillo-code.git
  GIT_TAG        carma_10.4.x
  SOURCE_DIR ${PROJECT_SOURCE_DIR}/extern/armadillo-code
)

FetchContent_GetProperties(armadillo_fork)

STRING(TOLOWER "ArmadilloFork" lcName)
IF (NOT ${lcName}_POPULATED)
  # Fetch the content using previously declared details
  FetchContent_Populate(ArmadilloFork)
ENDIF ()
