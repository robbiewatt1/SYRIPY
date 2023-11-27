cmake_minimum_required(VERSION 3.18 FATAL_ERROR)
project(track-example)

find_package(Torch REQUIRED)
set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} ${TORCH_CXX_FLAGS}")

include_directories("/home/rwatt/Documents/libTorch/test/src")

add_executable(track-example track-example.cpp
src/Field.cpp
src/PtTrack.cpp)



target_link_libraries(track-example "${TORCH_LIBRARIES}")
set_property(TARGET track-example PROPERTY CXX_STANDARD 17)
target_compile_definitions(track-example PRIVATE USE_TORCH=1)
