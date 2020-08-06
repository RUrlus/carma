#!/usr/bin/env bash
set -eo pipefail
[[ "$DEBUG_CI" == true ]] && set -x

case $TRAVIS_OS_NAME in
  linux|osx)
    TARGET=all
    ;;
  windows)
    TARGET=ALL_BUILD
    ;;
  *)
    echo "Unknown OS [$TRAVIS_OS_NAME]"
    exit 1
    ;;
esac

# CMake configuration
${TRAVIS_BUILD_DIR}/.travis-ci/configure.sh

cd build

cmake --build . --target ${TARGET} --config ${CMAKE_BUILD_TYPE}

if [[ "$COVERAGE" == "true" ]]; then
  cmake --build . --target coverage --config ${CMAKE_BUILD_TYPE}
else
  ctest -C ${CMAKE_BUILD_TYPE} -V
fi
