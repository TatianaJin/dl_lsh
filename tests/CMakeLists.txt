include_directories(${PROJECT_SOURCE_DIR} ${EXTERNAL_INCLUDE})

# Unit Test
file(GLOB_RECURSE UNITTEST_FILES ${PROJECT_SOURCE_DIR}/*_test.cpp) 
include_directories(${GTEST_INCLUDE})
add_executable(DLLSHUnitTest ${UNITTEST_FILES} test_main.cpp)
add_dependencies(DLLSHUnitTest gtest)
target_link_libraries(DLLSHUnitTest dl-lsh)
target_link_libraries(DLLSHUnitTest ${EXTERNAL_LIB})
target_link_libraries(DLLSHUnitTest ${GTEST_LIBRARIES})
target_link_libraries(DLLSHUnitTest ${GMOCK_LIBRARIES})
default_prop(DLLSHUnitTest)
add_dependencies(DLLSHUnitTest ${external_project_dependencies})
