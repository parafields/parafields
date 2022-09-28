cmake_minimum_required(VERSION 3.20)

# Set a name and version number for the project
project(
  parafields
  VERSION 0.1
  LANGUAGES C CXX)

# We allow <Package>_ROOT (env) variables for locating dependencies
cmake_policy(SET CMP0074 NEW)

# Define user-facing options
option(FORCE_SEQUENTIAL "Force a sequential build" OFF)
option(BUILD_SINGLE_PRECISION "Enable building single precision support" OFF)
option(BUILD_DOUBLE_PRECISION "Enable building double precision support" ON)

# Enable PIC for Python bindings
set(CMAKE_POSITION_INDEPENDENT_CODE ON)

# Define the minimum C++ standard that is required
set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

# Configure and build our external dependencies
configure_file(cmake/external.cmake.in external/CMakeLists.txt @ONLY)
execute_process(
  COMMAND ${CMAKE_COMMAND} -G "${CMAKE_GENERATOR}" .
  WORKING_DIRECTORY ${CMAKE_BINARY_DIR}/external COMMAND_ERROR_IS_FATAL ANY)
execute_process(COMMAND ${CMAKE_COMMAND} --build .
                WORKING_DIRECTORY ${CMAKE_BINARY_DIR}/external)

# Find the parafields library
set(CMAKE_PREFIX_PATH
    ${CMAKE_BINARY_DIR}/parafields-core-inst ${CMAKE_BINARY_DIR}/dune-inst
    ${CMAKE_BINARY_DIR}/fftw3-inst ${CMAKE_BINARY_DIR}/fakempi-inst
    "${CMAKE_PREFIX_PATH}")

find_package(parafields REQUIRED)
find_package(pybind11 REQUIRED)

# Build the Python package
pybind11_add_module(_parafields src/parafields/_parafields.cpp)
target_link_libraries(_parafields PUBLIC parafields::parafields
                                         pybind11::pybind11)
target_include_directories(_parafields PUBLIC ${PROJECT_SOURCE_DIR}/include)
if(MPI4PY_INCLUDE_DIR)
  target_include_directories(_parafields PUBLIC ${MPI4PY_INCLUDE_DIR})
endif()

# Install the compiled modules in the correct location
install(TARGETS _parafields DESTINATION .)

# This prints a summary of found dependencies
include(FeatureSummary)
feature_summary(WHAT ALL)
