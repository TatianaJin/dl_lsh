cmake_minimum_required(VERSION 3.5.0)
project(dl-lsh)

# set standard
set(CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_STANDARD 14)
# set output directory
set(CMAKE_RUNTIME_OUTPUT_DIRECTORY ${PROJECT_BINARY_DIR})

if(CMAKE_COMPILER_IS_GNUCXX)
    set(CMAKE_CXX_FLAGS_DEBUG "-g")
    set(CMAKE_CXX_FLAGS_RELWITHDEBINFO "-O2 -g")
    set(CMAKE_CXX_FLAGS_RELEASE "-O3 -w")
    set(CMAKE_CXX_FLAGS ${CMAKE_CXX_FLAGS} "-Wno-deprecated-declarations")
endif()

# external dependencies
include(cmake/dep.cmake)
include(cmake/utils.cmake)
include(cmake/modules/glog.cmake)
include(cmake/modules/gflags.cmake)
include(cmake/modules/gtest.cmake)

# external includes
set (EXTERNAL_INCLUDE ${EIGEN_INCLUDE_DIR} ${GLOG_INCLUDE_DIR} ${GFLAGS_INCLUDE_DIR})
# external libraries
find_package(Threads)
set(EXTERNAL_LIB ${GLOG_LIBRARY} ${GFLAGS_LIBRARY} ${CMAKE_THREAD_LIBS_INIT})


# all the components
add_subdirectory(dataset)
add_subdirectory(lsh)
add_subdirectory(nn)
add_subdirectory(exp)
add_subdirectory(tests)

# libdl-lsh.a
set(dl-lsh-objs $<TARGET_OBJECTS:dataset-objs> $<TARGET_OBJECTS:nn-objs> $<TARGET_OBJECTS:lsh-objs>)
add_library(dl-lsh STATIC ${dl-lsh-objs})
