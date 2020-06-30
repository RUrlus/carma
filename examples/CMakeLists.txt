CMAKE_MINIMUM_REQUIRED(VERSION 3.14)

SET(MODNAME "example_carma")

# ##############################################################################
#                                  EXECUTABLE                                  #
# ##############################################################################
pybind11_add_module(${MODNAME}
    MODULE
        arraystore.cpp
        example_bindings.cpp
        manual_conversion.cpp
        automatic_conversion.cpp
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
INSTALL(TARGETS ${MODNAME} DESTINATION examples)
FILE(GLOB PY_TEST_FILES "${PROJECT_SOURCE_DIR}/examples/*.py")
INSTALL(FILES ${PY_TEST_FILES} DESTINATION examples)
