include_directories(${PROJECT_SOURCE_DIR} ${EXTERNAL_INCLUDE})

file(GLOB nn-src-files
    utils.cpp
    HiddenLayer.cpp
    ReLUNeuronLayer.cpp
    )

cache_var(nn-src-files ${nn-src-files})
add_library(nn-objs OBJECT ${nn-src-files})
default_prop(nn-objs)

add_dependencies(nn-objs ${external_project_dependencies})
