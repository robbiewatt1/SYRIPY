cmake_minimum_required(VERSION 3.5)

project(cTrack)


if (BUILD_LIBTORCH)
    execute_process(COMMAND ${PYTHON_EXECUTABLE} -c "import torch; print(torch.utils.cmake_prefix_path)"
    RESULT_VARIABLE result
    OUTPUT_VARIABLE CMAKE_PREFIX_PATH
    OUTPUT_STRIP_TRAILING_WHITESPACE)
    list(APPEND CMAKE_PREFIX_PATH "${_tmp_dir2}")
    find_package(Torch REQUIRED)
    set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} ${TORCH_CXX_FLAGS} -fopenmp -O3")
else()
    set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -fopenmp -O3")
endif()


if(SKBUILD)
  execute_process(
    COMMAND
      python3 -c
      "import pybind11; print(pybind11.get_cmake_dir())"
    OUTPUT_VARIABLE _tmp_dir
    OUTPUT_STRIP_TRAILING_WHITESPACE COMMAND_ECHO STDOUT)
  list(APPEND CMAKE_PREFIX_PATH "${_tmp_dir}")
  find_package(pybind11 CONFIG REQUIRED)
endif()



pybind11_add_module(cTrack MODULE  SYRIPY/Tracking/cTrack/Bindings.cpp
 SYRIPY/Tracking/cTrack/Field.cpp SYRIPY/Tracking/cTrack/Track.cpp )

 if (BUILD_LIBTORCH)
 target_link_libraries(cTrack PRIVATE ${Python_LIBRARIES} ${TORCH_LIBRARIES})
 target_compile_definitions(cTrack PRIVATE USE_TORCH=1)
endif()

install(TARGETS cTrack LIBRARY DESTINATION .)