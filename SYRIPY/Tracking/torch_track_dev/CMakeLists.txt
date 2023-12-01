cmake_minimum_required(VERSION 3.18 FATAL_ERROR)
project(cTrack)

if (BUILD_LIBTORCH)
    execute_process(COMMAND python3 -c "import torch; print(torch.utils.cmake_prefix_path)"
    RESULT_VARIABLE result
    OUTPUT_VARIABLE CMAKE_PREFIX_PATH
    OUTPUT_STRIP_TRAILING_WHITESPACE)
    message(CMAKE_PREFIX_PATH)
    find_package(Torch REQUIRED)
    set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} ${TORCH_CXX_FLAGS} -O3")
else()
    set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS}")
endif()


find_package(pybind11 CONFIG REQUIRED)

include_directories(${Python_INCLUDE_DIRS})

add_library(pybind11 INTERFACE)

pybind11_add_module(cTrack src/Bindings.cpp src/Field.cpp
    src/Track.cpp)
if (BUILD_LIBTORCH)
    target_link_libraries(cTrack PRIVATE ${Python_LIBRARIES} ${TORCH_LIBRARIES})
    target_compile_definitions(cTrack PRIVATE USE_TORCH=1)
else()
    target_link_libraries(cTrack PRIVATE ${Python_LIBRARIES})
endif()