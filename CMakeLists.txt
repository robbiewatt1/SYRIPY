cmake_minimum_required(VERSION 3.4)

project(cTrack)

if(SKBUILD)
  # Scikit-Build does not add your site-packages to the search path automatically,
  # so we need to add it _or_ the pybind11 specific directory here.
  execute_process(
    COMMAND
      "${PYTHON_EXECUTABLE}" -c
      "import pybind11; print(pybind11.get_cmake_dir())"
    OUTPUT_VARIABLE _tmp_dir
    OUTPUT_STRIP_TRAILING_WHITESPACE COMMAND_ECHO STDOUT)
  list(APPEND CMAKE_PREFIX_PATH "${_tmp_dir}")
endif()

set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -O3 -fopenmp")

find_package (Eigen3 3.3 REQUIRED NO_MODULE)
find_package(pybind11 CONFIG REQUIRED)

pybind11_add_module(cTrack MODULE  ERS/Tracking/cTrack/Bindings.cpp
 ERS/Tracking/cTrack/Field.cpp ERS/Tracking/cTrack/Track.cpp )
target_link_libraries(cTrack PRIVATE Eigen3::Eigen)

install(TARGETS cTrack LIBRARY DESTINATION .)
