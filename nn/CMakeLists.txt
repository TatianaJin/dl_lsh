include_directories(${PROJECT_SOURCE_DIR} ${EXTERNAL_INCLUDE})

file(GLOB nn-src-files
    HiddenLayer.cpp
    NeuralCoordinator.cpp
    NeuronLayer.cpp
    ReLUNeuronLayer.cpp
    ICostFunction.cpp
    NeuralNetwork.cpp
    NN_parameters.cpp
    SoftMaxNeuronLayer.cpp)
cache_var(nn-src-files ${nn-src-files})
add_library(nn-objs OBJECT ${nn-src-files})
default_prop(nn-objs)
