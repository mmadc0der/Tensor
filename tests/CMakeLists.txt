cmake_minimum_required(VERSION 3.20)
find_package(Threads REQUIRED)

if (TENSOR_ENABLE_TESTS)
    add_executable(tensor_tests
        unit/tensor_core_test.cpp
        unit/api_factories_test.cpp
        unit/views_test.cpp
    )
    target_link_libraries(tensor_tests PRIVATE
        tensor
        GTest::gtest
        GTest::gtest_main
        Threads::Threads
    )
    if (MSVC)
        target_compile_options(tensor_tests PRIVATE /W4 /permissive- /EHsc)
    else()
        target_compile_options(tensor_tests PRIVATE -Wall -Wextra -Wpedantic)
    endif()

    add_test(NAME tensor_unit_tests COMMAND tensor_tests)
    set_tests_properties(tensor_unit_tests PROPERTIES LABELS "unit")
endif()

if (TENSOR_ENABLE_BENCHMARKS)
    add_executable(tensor_bench
        bench/tensor_bench.cpp
    )
    target_link_libraries(tensor_bench PRIVATE tensor benchmark::benchmark Threads::Threads)
    if (MSVC)
        target_compile_options(tensor_bench PRIVATE /W4 /permissive- /EHsc)
    else()
        target_compile_options(tensor_bench PRIVATE -Wall -Wextra -Wpedantic)
    endif()
endif()


