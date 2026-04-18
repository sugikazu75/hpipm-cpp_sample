cmake_minimum_required(VERSION 3.0.2)
project(hpipm-cpp_sample)

find_package(Eigen3 REQUIRED NO_MODULE)
find_package(catkin REQUIRED COMPONENTS
  hpipm-cpp
  )

catkin_package()

include_directories(
  ${catkin_INCLUDE_DIRS}
  ${EIGEN3_INCLUDE_DIRS}
  )

# add_executable(example_dense_qp example/example_dense_qp.cpp)
# target_link_libraries(example_dense_qp hpipm-cpp::hpipm-cpp)

add_executable(example_mpc example/example_mpc.cpp)
target_link_libraries(example_mpc ${catkin_LIBRARIES})

add_executable(example_ocp_qp example/example_ocp_qp.cpp)
target_link_libraries(example_ocp_qp ${catkin_LIBRARIES})
