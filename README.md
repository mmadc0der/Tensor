# **Tensor**

`Tensor` is a C++ tensor runtime project that is moving toward a CPU-first,
trainable core with Python bindings for flexible experimentation and testing.

The intended development style is:

- C++ first
- WSL/Linux first
- `clang++` or `g++` builds through CMake + Ninja
- pybind11 bindings for Python-side checks and quick experiments

## Current Focus

The near-term goal is not to build a huge framework in one shot. The project is
being shaped around a small, verifiable core:

- runtime tensor storage and views
- scalar tensor ops first
- minimal reverse-mode autograd
- a solid `Linear` path as the first trainable primitive
- Python bindings to inspect behavior from scripts

## Recommended Environment

The recommended primary environment is WSL Ubuntu or native Linux.

Suggested toolchain:

- `clang++` as the main compiler
- `g++` as a secondary compiler target
- CMake + Ninja
- Python 3 with development headers

On Ubuntu/WSL, start with:

```bash
sudo apt update
sudo apt install -y \
  build-essential \
  cmake \
  ninja-build \
  git \
  pkg-config \
  python3 \
  python3-dev \
  python3-venv \
  python3-pip
```

If you want a newer Clang than your distro default, install a versioned LLVM
toolchain and then point CMake presets at explicit binaries such as
`clang-20` / `clang++-20`.

## Python Environment

Python is mainly used here for:

- binding smoke tests
- numerical sanity checks
- quick scripting around the C++ library

Project-level Python packaging metadata now lives at the repository root, while
the importable Python code and tests stay under `python/`.

- `pyproject.toml`
- `requirements/`
- `python/tests/`
- `python/tensor/`
- `python/bindings.cpp`
- `python/CMakeLists.txt`

If you use `uv`, a practical flow is:

```bash
uv venv
uv sync --dev
uv pip install -e .
```

This installs the base Python tooling plus the default dev group, but does not
pull `torch`. The editable install now performs a packaged native build through
`pyproject.toml`, so `uv pip install -e .` builds and installs the local C++
extension instead of relying only on an external CMake build tree.

Extra groups:

- `uv sync --dev --group bench` to add benchmark tooling
- `uv sync --dev --group bench --group bench-torch` to add PyTorch-backed benchmark comparisons

The grouped `pip` requirements are also available under `requirements/`
for explicit layered installs.

## Verification Gates

The repo now has three named verification gates:

- `fast`: primary toolchain plus packaged Python tests
- `full`: GCC and Clang correctness checks plus packaged Python tests
- `release`: `full` plus a clean wheel-install smoke outside the repo tree

WSL/Linux helper scripts live under `scripts/`:

```bash
bash scripts/verify-fast.sh
bash scripts/verify-full.sh
bash scripts/verify-release.sh
```

`verify-full.sh` and `verify-release.sh` expect `clang++` to be available in
the active WSL/Linux environment.

The detailed command matrix, pass criteria, package smoke workflow, and
feature-based release roadmap live in `docs/verification.md`.

## Build Presets

The repo currently exposes CMake presets for GCC and Clang builds.

Example:

```bash
cmake --preset gcc-release
cmake --build --preset build-gcc-release

cmake --preset clang-release
cmake --build --preset build-clang-release
```

Useful options:

```bash
-DTENSOR_ENABLE_TESTS=ON
-DTENSOR_ENABLE_BENCHMARKS=ON
-DTENSOR_ENABLE_PYTHON=ON
-DTENSOR_ENABLE_PYTEST=ON
```

## Tests

### C++ unit tests

```bash
cmake --preset gcc-release-tests
cmake --build --preset build-gcc-release-tests
ctest --preset test-gcc-release -j
```

### Python extension and pytest

```bash
cmake --preset gcc-release-pytest
cmake --build --preset build-gcc-release-pytest --target tensor_py
ctest --preset pytest-gcc-release
```

From the repository root, the packaged Python install can be exercised directly:

```bash
uv pip install -e .
uv run python -m pytest -q -m "not benchmark"
```

Benchmark tests are separated from the normal Python test path and become active
when benchmark dependencies are installed.

To verify the installed package outside the repository tree:

```bash
bash scripts/package-smoke.sh
```

This builds a wheel, installs it into a fresh temporary environment, and runs
`examples/python/package_smoke.py` from outside the project directory.

### Benchmarks

```bash
cmake --preset gcc-release -DTENSOR_ENABLE_BENCHMARKS=ON
cmake --build --preset build-gcc-release -DTENSOR_ENABLE_BENCHMARKS=ON
./build/gcc-release/tensor_bench
```

## Repository Shape Today

The repo is in transition, but the current split is roughly:

- `src/tensor/`: core C++ tensor code
- `src/api/`: small public-facing helpers
- `python/`: Python package code, bindings, and Python-side tests
- `requirements/`: grouped Python dependency layers
- `pyproject.toml`: project-level packaging and editable build entrypoint
- `tests/unit/`: C++ tests
- `tests/bench/`: C++ benchmarks
- `python/tests/`: Python-side tests and benchmark comparisons

The long-term direction is to keep the C++ runtime as the main product and use
Python as a lightweight interface around it.

## Version Roadmap

Version lines are intended to represent stable capabilities, not just process:

- `0.2.x`: stable dense core centered on `matmul`
- `0.3.x`: minimal trainable path with autograd, `Linear`, one loss, and `SGD`
- `0.4.x`: reliable package usability outside the source tree
- `0.5.x`: careful model-building expansion beyond `Linear`

The detailed feature gates for each line are documented in
`docs/verification.md`.

## Windows Note

Windows support is currently most comfortable through WSL for the Clang +
Python-binding workflow.

MinGW/GCC on Windows can still be useful, but WSL is the preferred path if you
want a cleaner Linux-like build, newer Clang, and fewer Python extension
surprises.