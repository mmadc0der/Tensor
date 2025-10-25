cmake_minimum_required(VERSION 3.20)
project(Tensor LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 23)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)

# Output dir
set(CMAKE_RUNTIME_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR})

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
