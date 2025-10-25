cmake_minimum_required(VERSION 3.20)
project(Tensor LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 23)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)

# Output dir
set(CMAKE_RUNTIME_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR})

# Options
option(TENSOR_ENABLE_TESTS "Enable unit tests" OFF)
option(TENSOR_ENABLE_BENCHMARKS "Enable benchmarks" OFF)
option(TENSOR_ENABLE_PYTHON "Build Python module" OFF)
option(TENSOR_ENABLE_PYTEST "Enable pytest target (requires Python module)" OFF)

include(FetchContent)

if (TENSOR_ENABLE_TESTS)
    enable_testing()
    # Expect googletest available as a submodule under external/googletest
    add_subdirectory(external/googletest)
endif()

if (TENSOR_ENABLE_BENCHMARKS)
    # Google Benchmark
    FetchContent_Declare(
        googlebenchmark
        URL https://github.com/google/benchmark/archive/refs/tags/v1.8.3.zip
    )
    set(BENCHMARK_ENABLE_TESTING OFF CACHE BOOL "" FORCE)
    set(BENCHMARK_ENABLE_GTEST_TESTS OFF CACHE BOOL "" FORCE)
    FetchContent_MakeAvailable(googlebenchmark)
endif()

if (TENSOR_ENABLE_PYTHON)
    # pybind11
    FetchContent_Declare(
        pybind11
        URL https://github.com/pybind/pybind11/archive/refs/tags/v2.12.0.zip
    )
    FetchContent_MakeAvailable(pybind11)
    find_package(Python3 COMPONENTS Interpreter Development.Module REQUIRED)
endif()

add_library(tensor STATIC
    src/tensor/Tensor.cpp
    src/api/Api.hpp
)

# Public headers for consumers
target_include_directories(tensor
    PUBLIC
        ${CMAKE_CURRENT_SOURCE_DIR}/src
)

# Warnings and opts
if (MSVC)
    target_compile_options(tensor PRIVATE /W4 /permissive- /EHsc)
else()
    target_compile_options(tensor PRIVATE -Wall -Wextra -Wpedantic)
endif()

add_executable(tensor_example src/main.cpp)

target_link_libraries(tensor_example PRIVATE tensor)

# Subdirectories for tests and benchmarks
if (TENSOR_ENABLE_TESTS OR TENSOR_ENABLE_BENCHMARKS)
    add_subdirectory(tests)
endif()

# Python bindings
if (TENSOR_ENABLE_PYTHON)
    add_subdirectory(src/python)
endif()

# Pytest convenience target
if (TENSOR_ENABLE_PYTEST)
    if (NOT TENSOR_ENABLE_PYTHON)
        message(FATAL_ERROR "TENSOR_ENABLE_PYTEST requires TENSOR_ENABLE_PYTHON=ON")
    endif()
    find_package(Python3 COMPONENTS Interpreter REQUIRED)
    add_custom_target(pytest
        COMMAND ${CMAKE_COMMAND} -E env 
            PYTHONPATH=$<TARGET_FILE_DIR:tensor_py>
            ${Python3_EXECUTABLE} -m pytest -q
        WORKING_DIRECTORY ${CMAKE_CURRENT_SOURCE_DIR}/python
        DEPENDS tensor_py
    )
endif()
