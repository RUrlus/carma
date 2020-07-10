[[ "$DEBUG_CI" == true ]] && set -x

case "$COMPILER" in 
  gcc*)
    CXX=g++${COMPILER#gcc} 
    CC=gcc${COMPILER#gcc}
    ;;
  clang*)
    CXX=clang++${COMPILER#clang} 
    CC=clang${COMPILER#clang}
    # initially was only for clang ≥ 7
    CXXFLAGS="-stdlib=libc++"
    ;;
  msvc*)
    #echo "${COMPILER} not supported compiler"
    #exit 1
    ;;
  *)
    echo "${COMPILER} not supported compiler"
    exit 1
    ;;
esac

case "$DEBUG" in 
 true)
    CMAKE_BUILD_TYPE="Debug"
    ;;
 false|"")
     CMAKE_BUILD_TYPE="Release"
     ;;
  *)
    echo "Not supported DEBUG flag [$DEBUG]"
    exit 1
    ;;
esac

export CXX CC CXXFLAGS CMAKE_BUILD_TYPE

case $TRAVIS_OS_NAME in
  linux|osx)
    export PY_CMD=python$PYTHON
    export PATH=/usr/bin:$PATH
    ;;
  windows)
    export PATH=/c/Python37:$PATH
    export PY_CMD=python
    ;;
  *)
    echo "Unknown OS [$TRAVIS_OS_NAME]"
    exit 1
    ;;
esac
