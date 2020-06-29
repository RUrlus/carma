CMAKE_MINIMUM_REQUIRED(VERSION 3.14)

SET(MODNAME "test_carma")
SET(INCLUDE_BASE "${PROJECT_SOURCE_DIR}/tests/include")
SET(ARMA_INCLUDE "${PROJECT_SOURCE_DIR}/tests/include/armadillo/include")

SET(SUBPATH "${PROJECT_SOURCE_DIR}/tests/src")
SET(TESTPATH "${PROJECT_SOURCE_DIR}/tests/libs")

SET(PROJECT_RELEASEE_FLAGS "-march=native" "-mtune=native")
SET(PROJECT_RELEASEE_DEFINITIONS ARMA_NO_DEBUG)
# see linkg for a more expensive approach https://github.com/VcDevel/Vc/blob/master/cmake/OptimizeForArchitecture.cmake

# ##############################################################################
#                                  EXECUTABLE                                  #
# ##############################################################################
ADD_SUBDIRECTORY("${INCLUDE_BASE}/pybind11")
pybind11_add_module(${MODNAME}
    MODULE
        ${SUBPATH}/bindings.cpp
        ${SUBPATH}/test_store.cpp
        ${SUBPATH}/test_nparray.cpp
        ${SUBPATH}/test_arr_to_mat.cpp
        ${SUBPATH}/test_mat_to_arr.cpp
        ${SUBPATH}/test_type_caster.cpp
)

TARGET_INCLUDE_DIRECTORIES(${MODNAME}
    PUBLIC
        $<INSTALL_INTERFACE:include>
        $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/include>
    PRIVATE
        ${ARMA_INCLUDE}
)

TARGET_COMPILE_OPTIONS(${MODNAME}
    PUBLIC
        "$<$<CONFIG:RELEASE>:${PROJECT_RELEASEE_FLAGS}>"
)

TARGET_COMPILE_DEFINITIONS(${MODNAME}
    PUBLIC
        "$<$<CONFIG:RELEASE>:${PROJECT_RELEASE_DEFINITIONS}>"
)

# ##############################################################################
#                                   INSTALL                                    #
# ##############################################################################
INSTALL(TARGETS ${MODNAME} DESTINATION ${TESTPATH})
