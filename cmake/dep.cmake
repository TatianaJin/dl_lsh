set(EIGEN_FIND_REQUIRED true)
find_path(EIGEN_INCLUDE_DIR NAMES eigen3/Eigen/Dense)  # find in system variable PATH
if (EIGEN_INCLUDE_DIR)
    set(EIGEN_INCLUDE_DIR ${EIGEN_INCLUDE_DIR}/eigen3)
    message (STATUS "Found Eigen:")
    message (STATUS "   (Headers)   ${EIGEN_INCLUDE_DIR}")
else(EIGEN_INCLUDE_DIR)
    message(FATAL_ERROR "Could not find Eigen")
endif(EIGEN_INCLUDE_DIR)
