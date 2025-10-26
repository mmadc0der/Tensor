cmake_minimum_required(VERSION 3.20)

pybind11_add_module(tensor_py bindings.cpp)
target_link_libraries(tensor_py PRIVATE tensor)

if (MSVC)
    target_compile_options(tensor_py PRIVATE /W4 /permissive- /EHsc)
else()
    target_compile_options(tensor_py PRIVATE -Wall -Wextra -Wpedantic)
endif()

# On Windows with GCC/Clang (non-MSVC), link C++ runtime statically to avoid missing DLLs at import time
if (WIN32 AND CMAKE_CXX_COMPILER_ID STREQUAL "GNU")
    # Statically link libstdc++/libgcc only under GCC to avoid Clang/LLVM MinGW pthread link issues
    target_link_options(tensor_py PRIVATE -static-libstdc++ -static-libgcc)
endif()

# On MinGW/LLVM toolchains, ensure winpthread is linked via Threads::Threads
if (WIN32 AND NOT MSVC)
    find_package(Threads REQUIRED)
    target_link_libraries(tensor_py PRIVATE Threads::Threads)
endif()

# LLVM MinGW uses winpthread; ensure it is explicitly linked when using Clang
if (WIN32 AND NOT MSVC AND CMAKE_CXX_COMPILER_ID STREQUAL "Clang")
    # Ensure libstdc++ precedes winpthread to satisfy symbol resolution order
    target_link_libraries(tensor_py PRIVATE stdc++ winpthread)
endif()


