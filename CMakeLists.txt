cmake_minimum_required(VERSION 3.0.2)
project(hpipm-cpp_sample)

find_package(catkin REQUIRED)
find_package(Eigen3 REQUIRED)
find_package(blasfeo CONFIG REQUIRED)
find_package(hpipm CONFIG REQUIRED)
find_package(hpipm-cpp REQUIRED)

catkin_package(
#  INCLUDE_DIRS include
#  LIBRARIES hpipm-cpp_sample
#  CATKIN_DEPENDS other_catkin_pkg
#  DEPENDS system_lib
)

include_directories(
  # include
  ${catkin_INCLUDE_DIRS}
  ${EIGEN_INCLUDE_DIR}
  ${hpipm-cpp_INCLUDE_DIR}
  )

add_executable(example_dense_qp example/example_dense_qp.cpp)
target_link_libraries(example_dense_qp hpipm-cpp::hpipm-cpp)

add_executable(example_mpc example/example_mpc.cpp)
target_link_libraries(example_mpc hpipm-cpp::hpipm-cpp)

add_executable(example_ocp_qp example/example_ocp_qp.cpp)
target_link_libraries(example_ocp_qp hpipm-cpp::hpipm-cpp)
