#!/usr/bin/env bash
set -eo pipefail
[[ "$DEBUG_CI" == true ]] && set -x

CMAKE_EXTRA_ARGS+=" -DBUILD_TESTS=ON"

if [ -n "$CPP" ]; then CPPSTD=-std=c++$CPP; fi
if [ "$NOWND" = true ]; then CMAKE_EXTRA_ARGS+=" -DCARMA_DONT_REQUIRE_OWNDATA"; fi 
if [ "$VALGRIND" = true ]; then CMAKE_EXTRA_ARGS+=" -DVALGRIND_TEST_WRAPPER=on"; fi 

case $TRAVIS_OS_NAME in
  linux|osx)
    ;;
  windows)
    CMAKE_EXTRA_ARGS+=" -DCMAKE_GENERATOR_PLATFORM=x64"
    ;;
  *)
    echo "Unknown OS [$TRAVIS_OS_NAME]"
    exit 1
    ;;
esac

mkdir build
cd build
cmake ${CMAKE_EXTRA_ARGS} \
  -DCMAKE_BUILD_TYPE=${CMAKE_BUILD_TYPE} \
  -DPYBIND11_PYTHON_VERSION=$PYTHON \
  -DPYBIND11_CPP_STANDARD=$CPPSTD \
  ..