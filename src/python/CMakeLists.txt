cmake_minimum_required(VERSION 3.20)

pybind11_add_module(tensor_py bindings.cpp)
target_link_libraries(tensor_py PRIVATE tensor)

if (MSVC)
    target_compile_options(tensor_py PRIVATE /W4 /permissive- /EHsc)
else()
    target_compile_options(tensor_py PRIVATE -Wall -Wextra -Wpedantic)
endif()


