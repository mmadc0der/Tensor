# **Tensor**

A simple tensor library for deep learning workloads.

## Prerequisites

- CMake 3.20+
- Ninja
- GCC (recommended): GCC 11+ (tested with MinGW-w64 on Windows)
- Python 3.11/3.12 (Interpreter + Development headers)
- Optional: GoogleTest (comes as submodule), pybind11 (submodule or fetched)

## Build

```bash
# Recommended: use CMakePresets with Ninja + GCC
cmake --preset gcc-release
cmake --build --preset build-gcc-release

# Clang (experimental; unsupported on Windows MinGW for Python bindings)
cmake --preset clang-release
cmake --build --preset build-clang-release
```

## Options

```bash
-DTENSOR_ENABLE_TESTS=ON        # GoogleTest unit tests (ctest -L unit)
-DTENSOR_ENABLE_BENCHMARKS=ON   # Google Benchmark (run tensor_bench)
-DTENSOR_ENABLE_PYTHON=ON       # Build pybind11 module tensor_py
-DTENSOR_ENABLE_PYTEST=ON       # Add pytest target (requires Python)
```

## Unit tests

```bash
cmake --preset gcc-release-tests
cmake --build --preset build-gcc-release-tests
ctest --preset test-gcc-release -j
```

## Benchmarks

```bash
cmake --preset gcc-release -DTENSOR_ENABLE_BENCHMARKS=ON
cmake --build --preset build-gcc-release -DTENSOR_ENABLE_BENCHMARKS=ON
./build/gcc-release/tensor_bench
```

## Python module and pytest

```bash
# Build Python extension and run pytest (GCC)
cmake --preset gcc-release-pytest
cmake --build --preset build-gcc-release-pytest --target tensor_py
ctest --preset pytest-gcc-release
```

### Running pytest manually

From `python/` directory, a locally built module is auto-discovered by `tests/conftest.py`.

```bash
python -m pytest -q
```

Built artifact location (for reference): `build/<preset>/src/python/tensor_py.*.pyd` (Windows) or `.so` (Linux).

### Benchmarks (pytest)

Run only benchmark tests and sort by mean (recommended view):

```bash
python -m pytest -q -k bench --benchmark-sort=mean
```

Optional, add compact names and useful columns:

```bash
python -m pytest -q -k bench \
  --benchmark-name=short \
  --benchmark-columns=median,min,max,mean,stddev \
  --benchmark-sort=mean
```

Save and compare benchmark runs:

```bash
python -m pytest -q -k bench --benchmark-save=gcc
python -m pytest -q -k bench --benchmark-compare --benchmark-compare-fail=mean:5%
```

## Windows

- Recommended toolchain: GCC via MSYS2 MinGW-w64 + Ninja.
- The Python extension links libstdc++/libgcc statically under GCC to avoid missing DLLs.
- PyTest via CTest sets `PYTHONPATH` and augments `PATH` so the module and runtime DLLs are discoverable.
- Clang with MinGW on Windows is currently unsupported for the Python binding due to libstdc++/winpthread link issues. Use GCC or WSL.
- On WSL, run the same Linux commands above.

## Troubleshooting

- ImportError: DLL load failed when importing `tensor_py` on Windows
  - Ensure you built with the GCC presets above.
  - Run tests via `ctest --preset pytest-gcc-release` or from `python/` directory (`conftest.py` adjusts paths automatically).
  - If running Python outside CTest and `python/`, add `build/<preset>/src/python` to `PYTHONPATH`.