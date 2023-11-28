cmake_minimum_required(VERSION 3.18 FATAL_ERROR)
project(cTrack)

find_package(Torch REQUIRED)
find_package(Python COMPONENTS Interpreter Development REQUIRED)
find_package(pybind11 CONFIG REQUIRED)


set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} ${TORCH_CXX_FLAGS}")

include_directories(${Python_INCLUDE_DIRS})

add_library(pybind11 INTERFACE)

pybind11_add_module(cTrack src/Bindings.cpp src/Field.cpp
src/PtTrack.cpp)

target_link_libraries(cTrack PRIVATE ${Python_LIBRARIES} ${TORCH_LIBRARIES})
target_compile_definitions(cTrack PRIVATE USE_TORCH=1)

