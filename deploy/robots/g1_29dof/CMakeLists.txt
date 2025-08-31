cmake_minimum_required(VERSION 3.12)
project(g1_29dof_controller LANGUAGES CXX)

# Force C++17
set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)

# Tell CMake to use GCC 11 explicitly
set(CMAKE_CXX_COMPILER "/usr/bin/g++-11")
set(CMAKE_C_COMPILER "/usr/bin/gcc-11")

# Make sure the GCC 11 libstdc++ path is baked into the binary (no LD_LIBRARY_PATH needed)
set(CMAKE_BUILD_RPATH "/usr/lib/gcc/x86_64-linux-gnu/11")
set(CMAKE_INSTALL_RPATH "/usr/lib/gcc/x86_64-linux-gnu/11")

find_package(Boost REQUIRED COMPONENTS program_options)
find_package(yaml-cpp REQUIRED)
find_package(fmt REQUIRED) # safer than raw fmt in link_libraries

include_directories(
  /usr/include/eigen3
  /usr/local/include/ddscxx # dds
  /usr/local/include/iceoryx/v2.0.2 # dds
  include
  ${PROJECT_SOURCE_DIR}/../../thirdparty/ocs2/include
  ${PROJECT_SOURCE_DIR}/../../thirdparty/onnxruntime-linux-x64-1.22.0/include
  ${PROJECT_SOURCE_DIR}/../../include/
)

# Avoid global link_libraries — use target_link_libraries for clarity
file(GLOB_RECURSE ADD_SRC_LIST
  ${PROJECT_SOURCE_DIR}/src/*.cpp
)

add_library(${PROJECT_NAME}_lib ${ADD_SRC_LIST})

add_executable(g1_ctrl main.cpp)

target_link_libraries(g1_ctrl PRIVATE
  ${PROJECT_NAME}_lib
  unitree_sdk2 ddsc ddscxx
  Boost::program_options yaml-cpp fmt::fmt
  pthread rt
  ${PROJECT_SOURCE_DIR}/../../thirdparty/onnxruntime-linux-x64-1.22.0/lib/libonnxruntime.so.1.22.0
)

