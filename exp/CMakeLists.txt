include_directories(${PROJECT_SOURCE_DIR} ${EXTERNAL_INCLUDE})

# MNIST Example
add_executable(MNISTExample mnist_example.cpp)
target_link_libraries(MNISTExample dl-lsh)
target_link_libraries(MNISTExample ${EXTERNAL_LIB})
default_prop(MNISTExample)
