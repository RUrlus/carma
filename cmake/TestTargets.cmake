IF (ENABLE_COVERAGE)
    # --coverage option is used to compile and link code instrumented for coverage analysis.
    # The option is a synonym for -fprofile-arcs -ftest-coverage (when compiling) and -lgcov (when linking).
    SET(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} --coverage")
    SET(CMAKE_C_FLAGS "${CMAKE_C_FLAGS} --coverage")
    SET(CMAKE_EXE_LINKER_FLAGS "${CMAKE_EXE_LINKER_FLAGS} --coverage")
    SET(CMAKE_SHARED_LINKER_FLAGS "${CMAKE_SHARED_LINKER_FLAGS} --coverage")
ENDIF ()

IF (VALGRIND_TEST_WRAPPER)
    IF (WIN32)
        MESSAGE(FATAL_ERROR "VALGRIND_TEST_WRAPPER is not supported in WIN32 environment")
    ENDIF ()
    FIND_PROGRAM(VALGRIND_EXECUTABLE valgrind)
    IF (NOT VALGRIND_EXECUTABLE)
        MESSAGE(FATAL_ERROR "VALGRIND_TEST_WRAPPER is requested but valgrind executable cannot be found")
    ENDIF ()
    MESSAGE(STATUS "Valgrind test mode enabled")
ENDIF ()

# ##############################################################################
#                               Clang-format                                   #
# ##############################################################################
## search for clang-format and add target
IF(NOT DEFINED CARMA_DEV_TARGET OR CARMA_DEV_TARGET)
    FIND_PROGRAM(CLANG_FORMAT clang-format)
    IF (CLANG_FORMAT)
        EXEC_PROGRAM(
            ${CLANG_FORMAT} ARGS -version OUTPUT_VARIABLE CLANG_FORMAT_RAW_VERSION
        )
        STRING(REGEX MATCH "[1-9][0-9]*\\.[0-9]+\\.[0-9]+"
                CLANG_FORMAT_VERSION ${CLANG_FORMAT_RAW_VERSION})
        IF (CLANG_FORMAT_VERSION VERSION_GREATER_EQUAL "6.0.0") 
            ADD_CUSTOM_TARGET(clang-format
                COMMAND echo "running ${CLANG_FORMAT} ..."
                COMMAND ${CMAKE_COMMAND}
                -DPROJECT_SOURCE_DIR="${PROJECT_SOURCE_DIR}"
                -DCLANG_FORMAT="${CLANG_FORMAT}"
                -P ${PROJECT_SOURCE_DIR}/cmake/ClangFormatProcess.cmake
            )
            MESSAGE(STATUS "clang-format target for updating code format is available")
        ELSE ()
            MESSAGE(WARNING "incompatible clang-format found (<6.0.0); clang-format target is not available.")
            ADD_CUSTOM_TARGET(clang-format
                COMMAND ${CMAKE_COMMAND} -E echo ""
                COMMAND ${CMAKE_COMMAND} -E echo "*** code formatting not available since clang-format version is incompatible ***"
                COMMAND ${CMAKE_COMMAND} -E echo ""
                COMMENT "Inform about not available code format."
            )
        ENDIF()
    ELSE ()
        MESSAGE(WARNING "clang-format no found; clang-format target is not available.")
        ADD_CUSTOM_TARGET(clang-format
            COMMAND ${CMAKE_COMMAND} -E echo ""
            COMMAND ${CMAKE_COMMAND} -E echo "*** code formatting not available since clang-format has not been found ***"
            COMMAND ${CMAKE_COMMAND} -E echo ""
            COMMENT "Inform about not available code format."
        )
    ENDIF ()

    #------------------------------------------------------
    # unit tests coverage

    IF (ENABLE_COVERAGE)
        FIND_PROGRAM(LCOV lcov)
        IF (NOT LCOV)
            MESSAGE(FATAL_ERROR "lcov not found, cannot perform coverage.")
        ENDIF ()
    
        # coveralls.io does not support striped paths
        #find_program (SED NAMES sed)
        #if (NOT SED)
        #    message(FATAL_ERROR "Unable to find sed")
        #else()
        #    # message(STATUS "sed found at ${SED}")
        #endif (NOT SED)
    
        # Don't forget '' around each pattern
        SET(LCOV_EXCLUDE_PATTERN "'${PROJECT_SOURCE_DIR}/third_party/*'")
    
        ADD_CUSTOM_TARGET(coverage
            # Cleanup previously generated profiling data
            COMMAND ${LCOV} --base-directory ${PROJECT_SOURCE_DIR} --directory ${PROJECT_BINARY_DIR} --zerocounters
            # Initialize profiling data with zero coverage for every instrumented line of the project
            # This way the percentage of total lines covered will always be correct, even when not all source code files were loaded during the test(s)
            COMMAND ${LCOV} --base-directory ${PROJECT_SOURCE_DIR} --directory ${PROJECT_BINARY_DIR} --capture --initial --output-file coverage_base.info
            # Run tests
            COMMAND ${CMAKE_CTEST_COMMAND} -j ${PROCESSOR_COUNT}
            # Collect data from executions
            COMMAND ${LCOV} --base-directory ${PROJECT_SOURCE_DIR} --directory ${PROJECT_BINARY_DIR} --capture --output-file coverage_ctest.info
            # Combine base and ctest results
            COMMAND ${LCOV} --add-tracefile coverage_base.info --add-tracefile coverage_ctest.info --output-file coverage_full.info
            # Extract only project data (--no-capture or --remove options may be used to select collected data)
            COMMAND ${LCOV} --remove coverage_full.info ${LCOV_EXCLUDE_PATTERN} --output-file coverage_filtered.info
            COMMAND ${LCOV} --extract coverage_filtered.info '${PROJECT_SOURCE_DIR}/*' --output-file coverage.info
            # coveralls.io does not support striped paths
            #COMMAND ${SED} -i.bak 's|SF:${PROJECT_SOURCE_DIR}/|SF:|g' coverage.info
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
ENDIF ()

SET(CMAKE_MODULE_PATH ${CMAKE_MODULE_PATH_SAVED})
UNSET(CMAKE_MODULE_PATH_SAVED)
