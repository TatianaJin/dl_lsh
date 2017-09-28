cmake_minimum_required(VERSION 3.5.0)
project(dl-lsh)

# set standard
set(CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_STANDARD 14)
# set output directory
set(CMAKE_RUNTIME_OUTPUT_DIRECTORY ${PROJECT_BINARY_DIR})


# external dependencies
include(cmake/dep.cmake)
include(cmake/utils.cmake)
# external includes
set (EXTERNAL_INCLUDE ${EIGEN_INCLUDE_DIR})
# external libraries
find_package(Threads)
set(EXTERNAL_LIB ${CMAKE_THREAD_LIBS_INIT})


# all the components
add_subdirectory(dataset)
add_subdirectory(lsh)
add_subdirectory(nn)
add_subdirectory(exp)
# libdl-lsh.a
set(dl-lsh-objs $<TARGET_OBJECTS:dataset-objs> $<TARGET_OBJECTS:nn-objs> $<TARGET_OBJECTS:lsh-objs>)
add_library(dl-lsh STATIC ${dl-lsh-objs})
