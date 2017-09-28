include_directories(${PROJECT_SOURCE_DIR} ${EXTERNAL_INCLUDE})

file(GLOB dataset-src-files
    MNISTDataSet.cpp)
cache_var(dataset-src-files ${dataset-src-files})
add_library(dataset-objs OBJECT ${dataset-src-files})
default_prop(dataset-objs)
