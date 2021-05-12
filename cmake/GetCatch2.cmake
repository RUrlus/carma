include(FetchContent)

FetchContent_Declare(
  Catch2
  GIT_REPOSITORY https://github.com/catchorg/Catch2.git
  GIT_TAG        v2.13.6
  SOURCE_DIR ${PROJECT_SOURCE_DIR}/extern/Catch2
)
FetchContent_MakeAvailable(Catch2)
