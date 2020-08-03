CMAKE_MINIMUM_REQUIRED(VERSION 3.7)

SET(NATIVE_MODNAME "test_carma_native")

# ##############################################################################
#                                   C++ TESTS                                  #
# ##############################################################################
ADD_EXECUTABLE(${NATIVE_MODNAME}
    src/test_main.cpp
    src/test_mat_to_arr_native.cpp
)

TARGET_LINK_LIBRARIES(${NATIVE_MODNAME} PUBLIC carma Catch2::Catch2 pybind11::embed)
    
TARGET_INCLUDE_DIRECTORIES(${NATIVE_MODNAME}
    PUBLIC
        $<INSTALL_INTERFACE:include>
        $<BUILD_INTERFACE:${PROJECT_SOURCE_DIR}/python/include>
)

INCLUDE(CTest)
INCLUDE(${PROJECT_SOURCE_DIR}/third_party/Catch2/contrib/Catch.cmake)
catch_discover_tests(${NATIVE_MODNAME})

INSTALL(TARGETS ${NAIVE_MODNAME} RUNTIME DESTINATION tests)

# ##############################################################################
#                                 PYTHON TESTS                                 #
# ##############################################################################
SET(MODNAME "test_carma")

# ******************************************************************************
#                                  EXECUTABLE                                  *
# ******************************************************************************
pybind11_add_module(${MODNAME}
    MODULE
        src/bindings.cpp
        # src/test_store.cpp
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

# ******************************************************************************
#                                   INSTALL                                    *
# ******************************************************************************
INSTALL(TARGETS ${MODNAME} DESTINATION tests)
FILE(GLOB PY_TEST_FILES "${PROJECT_SOURCE_DIR}/tests/*.py")
INSTALL(FILES ${PY_TEST_FILES} DESTINATION tests)

# ******************************************************************************
#                                    TESTS                                     *
# ******************************************************************************
if (VALGRIND_TEST_WRAPPER)
    SET(VALGRIND_LOG_FILE /tmp/valgrind-output)
    ADD_TEST(NAME pytest
            WORKING_DIRECTORY ${CMAKE_CURRENT_SOURCE_DIR}
            COMMAND ${VALGRIND_EXECUTABLE} --show-leak-kinds=definite --log-file=${VALGRIND_LOG_FILE} ${PYTHON_EXECUTABLE} -m pytest -vv --valgrind --valgrind-log=${VALGRIND_LOG_FILE})
    SET_PROPERTY(TEST pytest APPEND PROPERTY ENVIRONMENT "PYTHONMALLOC=malloc")
else()
    ADD_TEST(NAME pytest
             WORKING_DIRECTORY ${CMAKE_CURRENT_SOURCE_DIR}
             COMMAND ${PYTHON_EXECUTABLE} -m pytest)
endif()

if (WIN32)
    SET_PROPERTY(TEST pytest
            APPEND PROPERTY ENVIRONMENT "PYTHONPATH=${CMAKE_CURRENT_BINARY_DIR}/$<CONFIG>")
else()
    SET_PROPERTY(TEST pytest
            APPEND PROPERTY ENVIRONMENT "PYTHONPATH=${CMAKE_CURRENT_BINARY_DIR}")
endif()
