# **Tensor**

A simple tensor library for deep learning workloads.

## Build

```
# Recommended: use CMakePresets with Ninja + GCC/Clang
cmake --preset gcc-release
cmake --build --preset build-gcc-release

# or Clang
cmake --preset clang-release
cmake --build --preset build-clang-release
```

## Options

```
-DTENSOR_ENABLE_TESTS=ON        # GoogleTest unit tests (ctest -L unit)
-DTENSOR_ENABLE_BENCHMARKS=ON   # Google Benchmark (run tensor_bench)
-DTENSOR_ENABLE_PYTHON=ON       # Build pybind11 module tensor_py
-DTENSOR_ENABLE_PYTEST=ON       # Add pytest target (requires Python)
```

## Unit tests

```
cmake --preset gcc-release-tests
cmake --build --preset build-gcc-release-tests
ctest --preset test-gcc-release -j
```

## Benchmarks

```
cmake -S . -B build -DTENSOR_ENABLE_BENCHMARKS=ON -DCMAKE_BUILD_TYPE=Release
cmake --build build --target tensor_bench
./build/tensor_bench
```

## Python module and pytest

```
# Build Python extension and run pytest
cmake --preset gcc-release-pytest
cmake --build --preset build-gcc-release-pytest --target tensor_py
ctest --preset pytest-gcc-release
```

## Windows

- Use Clang or GCC via MSYS2/WSL with Ninja generator.
- MSVC and MinGW toolchains are not required; prefer Clang/LLD or GCC.
- On WSL, run the same Linux commands above.