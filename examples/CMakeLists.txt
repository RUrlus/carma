CMAKE_MINIMUM_REQUIRED(VERSION 3.14)

SET(MODNAME "test_carma")
SET(TPATH "${PROJECT_SOURCE_DIR}/examples")

# ##############################################################################
#                                  EXECUTABLE                                  #
# ##############################################################################
pybind11_add_module(carma_examples
    MODULE
        ${PROJECT_SOURCE_DIR}/examples/arraystore.cpp
        ${PROJECT_SOURCE_DIR}/examples/example_bindings.cpp
        ${PROJECT_SOURCE_DIR}/examples/manual_conversion.cpp
        ${PROJECT_SOURCE_DIR}/examples/automatic_conversion.cpp
)

TARGET_INCLUDE_DIRECTORIES(carma_examples
    PUBLIC
        $<INSTALL_INTERFACE:include>
        $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/include>
    PRIVATE
        ${ARMA_INCLUDE}
)

TARGET_COMPILE_OPTIONS(carma_examples
    PUBLIC
        "$<$<CONFIG:RELEASE>:${PROJECT_RELEASE_FLAGS}>"
)

TARGET_COMPILE_DEFINITIONS(carma_examples
    PUBLIC
        "$<$<CONFIG:RELEASE>:${PROJECT_RELEASE_DEFINITIONS}>"
)

# ##############################################################################
#                                   INSTALL                                    #
# ##############################################################################
INSTALL(TARGETS carma_examples DESTINATION ${TPATH})
