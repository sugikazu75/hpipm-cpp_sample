cmake_minimum_required(VERSION 3.0.2)
project(hpipm-cpp_sample)

find_package(hpipm-cpp REQUIRED CONFIG)

# add_executable(example_dense_qp example/example_dense_qp.cpp)
# target_link_libraries(example_dense_qp hpipm-cpp::hpipm-cpp)

add_executable(example_mpc example/example_mpc.cpp)
target_link_libraries(example_mpc hpipm-cpp::hpipm-cpp)

add_executable(example_ocp_qp example/example_ocp_qp.cpp)
target_link_libraries(example_ocp_qp hpipm-cpp::hpipm-cpp)

install(
  TARGETS example_mpc example_ocp_qp
  RUNTIME DESTINATION bin
)
