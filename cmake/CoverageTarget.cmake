# ##############################################################################
#                                  COVERAGE                                    #
# ##############################################################################
IF (ENABLE_COVERAGE)
    FIND_PROGRAM(LCOV lcov)
    IF (NOT LCOV)
        MESSAGE(FATAL_ERROR "lcov not found, cannot perform coverage.")
    ENDIF ()

    SET(LCOV_EXCLUDE_PATTERN
        "'${PROJECT_SOURCE_DIR}/third_party/*'"
        "'${PROJECT_SOURCE_DIR}/extern/*'"
    )

    ADD_CUSTOM_TARGET(coverage
        COMMAND ${LCOV} --base-directory ${PROJECT_SOURCE_DIR} --directory ${PROJECT_BINARY_DIR} --zerocounters

        # Initial capture
        COMMAND ${LCOV} --base-directory ${PROJECT_SOURCE_DIR}
                        --directory ${PROJECT_BINARY_DIR}
                        --capture
                        --initial
                        --ignore-errors mismatch
                        --output-file coverage_base.info

        COMMAND ${CMAKE_CTEST_COMMAND} -j ${PROCESSOR_COUNT}

        # Post-test capture
        COMMAND ${LCOV} --base-directory ${PROJECT_SOURCE_DIR}
                        --directory ${PROJECT_BINARY_DIR}
                        --capture
                        --ignore-errors mismatch
                        --output-file coverage_ctest.info

        COMMAND ${LCOV} --add-tracefile coverage_base.info
                        --add-tracefile coverage_ctest.info
                        --output-file coverage_full.info

        COMMAND ${LCOV} --remove coverage_full.info ${LCOV_EXCLUDE_PATTERN} --output-file coverage_filtered.info
        COMMAND ${LCOV} --extract coverage_filtered.info '${PROJECT_SOURCE_DIR}/*' --output-file coverage.info

        DEPENDS tests
        COMMENT "Running test coverage."
        WORKING_DIRECTORY "${PROJECT_BINARY_DIR}"
    )
    MESSAGE(STATUS "coverage target for code coverage is available")

    FIND_PROGRAM(GENHTML genhtml)
    IF (NOT GENHTML)
        MESSAGE(WARNING "genhtml not found, cannot perform report-coverage.")
    ELSE ()
        ADD_CUSTOM_TARGET(coverage-report
            COMMAND ${CMAKE_COMMAND} -E remove_directory "${PROJECT_BINARY_DIR}/coverage"
            COMMAND ${CMAKE_COMMAND} -E make_directory "${PROJECT_BINARY_DIR}/coverage"
            COMMAND ${GENHTML} -o coverage -t "${CMAKE_PROJECT_NAME} test coverage" --ignore-errors source --legend --num-spaces 4 coverage.info
            COMMAND ${LCOV} --list coverage.info
            DEPENDS coverage
            COMMENT "Building coverage html report."
            WORKING_DIRECTORY "${PROJECT_BINARY_DIR}"
        )
    ENDIF ()
ELSE ()
    ADD_CUSTOM_TARGET(coverage
        COMMAND ${CMAKE_COMMAND} -E echo ""
        COMMAND ${CMAKE_COMMAND} -E echo "*** Use CMAKE_BUILD_TYPE=Coverage option in cmake configuration to enable code coverage ***"
        COMMAND ${CMAKE_COMMAND} -E echo ""
        COMMENT "Inform about not available code coverage."
    )
    ADD_CUSTOM_TARGET(coverage-report DEPENDS coverage)
ENDIF ()
