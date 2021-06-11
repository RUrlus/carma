IF (CMAKE_CXX_COMPILER_ID MATCHES "MSVC")
    # workaround like https://github.com/nlohmann/json/issues/1408
    # to avoid error like: carma\third_party\armadillo-code\include\armadillo_bits/arma_str.hpp(194):
    # error C2039: '_snprintf': is not a member of 'std' (compiling source file carma\tests\src\bindings.cpp) 
    ADD_DEFINITIONS(-DHAVE_SNPRINTF)
ENDIF ()
