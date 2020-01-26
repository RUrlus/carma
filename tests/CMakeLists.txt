CMAKE_MINIMUM_REQUIRED(VERSION 3.14.4)

SET(MODNAME "test_carma")
SET(SUBPATH "${PROJECT_SOURCE_DIR}/tests/src")
SET(TESTPATH "${PROJECT_SOURCE_DIR}/tests/libs")

# ##############################################################################
#                                  EXECUTABLE                                  #
# ##############################################################################
pybind11_add_module(${MODNAME}
    MODULE
        ${SUBPATH}/bindings.cpp
        ${SUBPATH}/test_nparray.cpp
        ${SUBPATH}/test_arr_to_mat.cpp
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
