cmake_minimum_required(VERSION 3.11)

# Set a name and version number for the project
project(
  parafields
  VERSION 0.1
  LANGUAGES C CXX)

# Configure and build our external dependencies
configure_file(external.cmake.in external/CMakeLists.txt)
execute_process(COMMAND ${CMAKE_COMMAND} -G "${CMAKE_GENERATOR}" .
                WORKING_DIRECTORY ${CMAKE_BINARY_DIR}/external)
execute_process(COMMAND ${CMAKE_COMMAND} --build .
                WORKING_DIRECTORY ${CMAKE_BINARY_DIR}/external)

# Find the parafields library
set(CMAKE_PREFIX_PATH
    ${CMAKE_BINARY_DIR}/core-inst ${CMAKE_BINARY_DIR}/dune-inst
    ${CMAKE_BINARY_DIR}/fftw3-inst "${CMAKE_PREFIX_PATH}")

find_package(parafields REQUIRED)

# Build the Python package
find_package(pybind11)
pybind11_add_module(_parafields src/parafields/_parafields.cpp)
target_link_libraries(_parafields PUBLIC parafields::parafields)

# Install the compiled modules in the correct location
install(TARGETS _parafields DESTINATION .)

# This prints a summary of found dependencies
include(FeatureSummary)
feature_summary(WHAT ALL)
