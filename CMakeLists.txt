cmake_minimum_required(VERSION 3.9.0)
project(stereye_cuda_tutorial LANGUAGES CXX CUDA) # enable cuda

set(CMAKE_BUILD_TYPE "RelWithDebInfo")
# set(CMAKE_CUDA_FLAGS ${CMAKE_CUDA_FLAGS}" -g -o3")
find_package ( OpenMP )

add_executable(example_01 src/example_01.cu)

add_library(cu_add src/example_02.h src/example_02.cu)
target_compile_features(cu_add PUBLIC cxx_std_11)

add_executable(example_02 src/example_02.cc)
target_link_libraries(example_02 cu_add)

add_executable(example_03 src/example_03.cu)

add_executable(example_04 src/example_04.cu)
target_compile_options(example_04 PRIVATE "-Xcompiler" ${OpenMP_CXX_FLAGS})
target_link_libraries(example_04 PRIVATE ${OpenMP_CXX_LIBRARIES})

add_executable(example_05 src/example_05.cu)

add_executable(example_06 src/example_06.cu)

add_executable(example_07 src/example_07.cu)
