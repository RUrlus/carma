CMAKE_MINIMUM_REQUIRED(VERSION 3.14)

SET(MODNAME "test_carma")

# ##############################################################################
#                                  EXECUTABLE                                  #
# ##############################################################################
pybind11_add_module(${MODNAME}
    MODULE
        src/bindings.cpp
        src/test_store.cpp
        src/test_nparray.cpp
        src/test_arr_to_mat.cpp
        src/test_mat_to_arr.cpp
        src/test_type_caster.cpp
)
TARGET_LINK_LIBRARIES(${MODNAME} PUBLIC carma)

TARGET_INCLUDE_DIRECTORIES(${MODNAME}
    PUBLIC
        $<INSTALL_INTERFACE:include>
        $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/include>
)

TARGET_COMPILE_OPTIONS(${MODNAME}
    PUBLIC
        "$<$<CONFIG:RELEASE>:${PROJECT_RELEASE_FLAGS}>"
)

TARGET_COMPILE_DEFINITIONS(${MODNAME}
    PUBLIC
        "$<$<CONFIG:RELEASE>:${PROJECT_RELEASE_DEFINITIONS}>"
)

# ##############################################################################
#                                   INSTALL                                    #
# ##############################################################################
INSTALL(TARGETS ${MODNAME} DESTINATION tests)
FILE(GLOB PY_TEST_FILES "${PROJECT_SOURCE_DIR}/tests/*.py")
INSTALL(FILES ${PY_TEST_FILES} DESTINATION tests)

# ##############################################################################
#                                    TESTS                                     #
# ##############################################################################
ADD_TEST(NAME pytest
         WORKING_DIRECTORY ${CMAKE_CURRENT_SOURCE_DIR}
         COMMAND ${Python3_EXECUTABLE} -m pytest)

if (WIN32)
    SET_PROPERTY(TEST pytest
            PROPERTY ENVIRONMENT "PYTHONPATH=${CMAKE_CURRENT_BINARY_DIR}/${CMAKE_BUILD_TYPE}")
else()
    SET_PROPERTY(TEST pytest
            PROPERTY ENVIRONMENT "PYTHONPATH=${CMAKE_CURRENT_BINARY_DIR}")
endif()
    