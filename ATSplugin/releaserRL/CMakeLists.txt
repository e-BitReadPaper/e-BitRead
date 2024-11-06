cmake_minimum_required(VERSION 3.0 FATAL_ERROR)
project(custom_ops)

find_package(Torch REQUIRED)

add_executable(releaserRL releaserRL.cpp)
target_link_libraries(replace "${TORCH_LIBRARIES}")
set_property(TARGET replace PROPERTY CXX_STANDARD 11)