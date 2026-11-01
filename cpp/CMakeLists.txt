cmake_minimum_required(VERSION 3.16)
project(cartpole_onnx CXX)
set(CMAKE_CXX_STANDARD 17)
if(NOT CMAKE_BUILD_TYPE)
  set(CMAKE_BUILD_TYPE Release)
endif()

# Path to the unpacked ONNX Runtime release, e.g. -DORT_DIR=$HOME/onnxruntime-linux-x64-1.22.0
set(ORT_DIR "" CACHE PATH "ONNX Runtime directory")
if(NOT ORT_DIR)
  message(FATAL_ERROR "Pass -DORT_DIR=/path/to/onnxruntime-linux-x64-<version>")
endif()

add_executable(cartpole_onnx cartpole_onnx.cpp)
target_include_directories(cartpole_onnx PRIVATE ${ORT_DIR}/include)
target_link_libraries(cartpole_onnx PRIVATE ${ORT_DIR}/lib/libonnxruntime.so)
set_target_properties(cartpole_onnx PROPERTIES BUILD_RPATH ${ORT_DIR}/lib)
