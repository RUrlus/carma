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

if [ -z ${PYTHON_SUFFIX+x} ]; then
  # variable not set, use PYTHON_VERSION
  export PYTHON_SUFFIX=${PYTHON_VERSION}
fi

if [ -n "$PYTHON_PREFIX_PATH" ]; then
  export PY_CMD=${PYTHON_PREFIX_PATH}/python$PYTHON_SUFFIX
else
  export PY_CMD=python$PYTHON_SUFFIX
fi
