include_directories(${PROJECT_SOURCE_DIR} ${EXTERNAL_INCLUDE})

file(GLOB lsh-src-files
        CosineDistance.cpp
        Histogram.cpp
        Pooling.cpp
        HashBuckets.cpp
        LSH.cpp
        RandomProjection.cpp
    )
cache_var(lsh-src-files ${lsh-src-files})
add_library(lsh-objs OBJECT ${lsh-src-files})
default_prop(lsh-objs)
