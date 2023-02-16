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

find_package(pybind11 CONFIG REQUIRED)
pybind11_add_module(cTrack MODULE  ERS/Tracking/cTrack/Bindings.cpp
 ERS/Tracking/cTrack/Field.cpp ERS/Tracking/cTrack/Track.cpp)

install(TARGETS cTrack LIBRARY DESTINATION .)
