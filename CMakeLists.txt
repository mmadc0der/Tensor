cmake_minimum_required(VERSION 3.20)
project(Tensor LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 23)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)

# Helpful for tooling (clangd, etc.)
set(CMAKE_EXPORT_COMPILE_COMMANDS ON)

# Output dir
set(CMAKE_RUNTIME_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR})

# Options
option(TENSOR_ENABLE_TESTS "Enable unit tests" OFF)
option(TENSOR_ENABLE_BENCHMARKS "Enable benchmarks" OFF)
option(TENSOR_ENABLE_PYTHON "Build Python module" OFF)
option(TENSOR_ENABLE_PYTEST "Enable pytest target (requires Python module)" OFF)

include(FetchContent)

# Enable CTest if either unit tests or pytest are requested
if (TENSOR_ENABLE_TESTS OR TENSOR_ENABLE_PYTEST)
    enable_testing()
endif()

if (TENSOR_ENABLE_TESTS)
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
    # pybind11: prefer local submodule, else fallback to FetchContent
    if (EXISTS ${CMAKE_CURRENT_SOURCE_DIR}/external/pybind11/CMakeLists.txt)
        add_subdirectory(external/pybind11)
    else()
        FetchContent_Declare(
            pybind11
            URL https://github.com/pybind/pybind11/archive/refs/tags/v2.12.0.zip
        )
        FetchContent_MakeAvailable(pybind11)
    endif()
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

# Build static library as PIC to allow linking into Python extension/shared libs
set_target_properties(tensor PROPERTIES POSITION_INDEPENDENT_CODE ON)

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
    # Prepare runtime PATH on Windows so Python can locate compiler runtime DLLs (e.g., libstdc++/libgcc)
    if (WIN32)
        get_filename_component(_CXX_DIR "${CMAKE_CXX_COMPILER}" DIRECTORY)
        # Build-time extra env for pytest command
        set(_PYTEST_ENV_EXTRA "PATH=${_CXX_DIR};$ENV{PATH}")
    endif()

    add_custom_target(pytest
        COMMAND ${CMAKE_COMMAND} -E env
            PYTHONPATH=$<TARGET_FILE_DIR:tensor_py>
            ${_PYTEST_ENV_EXTRA}
            ${Python3_EXECUTABLE} -m pytest -q
        WORKING_DIRECTORY ${CMAKE_CURRENT_SOURCE_DIR}/python
        DEPENDS tensor_py
    )
    # Also register pytest with CTest for integration with test presets
    add_test(NAME pytest COMMAND ${Python3_EXECUTABLE} -m pytest -q)
    set_tests_properties(pytest PROPERTIES WORKING_DIRECTORY "${CMAKE_CURRENT_SOURCE_DIR}/python")
    # Ensure Python can import built extension and find runtime libs
    set_tests_properties(pytest PROPERTIES ENVIRONMENT "PYTHONPATH=$<TARGET_FILE_DIR:tensor_py>")
    if (WIN32)
        set_property(TEST pytest APPEND PROPERTY ENVIRONMENT "PATH=${_CXX_DIR};$ENV{PATH}")
    endif()
    set_tests_properties(pytest PROPERTIES LABELS "py")
endif()
