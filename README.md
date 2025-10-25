# **Tensor**

A simple tensor library for deep learning workloads.

## Build

```
cmake -S . -B build -DCMAKE_BUILD_TYPE=Release
cmake --build build
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
cmake -S . -B build -DTENSOR_ENABLE_TESTS=ON
cmake --build build
ctest --test-dir build -L unit -j
```

## Benchmarks

```
cmake -S . -B build -DTENSOR_ENABLE_BENCHMARKS=ON -DCMAKE_BUILD_TYPE=Release
cmake --build build --target tensor_bench
./build/tensor_bench
```

## Python module and pytest

```
cmake -S . -B build -DTENSOR_ENABLE_PYTHON=ON -DTENSOR_ENABLE_PYTEST=ON
cmake --build build --target tensor_py
cmake --build build --target pytest
```