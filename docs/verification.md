# Verification Guide

This repo has three verification gates:

| Gate | When to run | Goal |
| --- | --- | --- |
| `fast` | Every meaningful code change | Catch regressions quickly on one primary toolchain |
| `full` | Before merge or review batch handoff | Recheck correctness across both supported Linux toolchains |
| `release` | Before tagging a minor or patch release | Prove the package works outside the source tree and run deeper manual checks |

The canonical environment is WSL/Linux. Benchmarks and Torch-backed comparisons remain opt-in.

## Fast Gate

Run this after changes to `src/`, `python/`, CMake files, packaging metadata, or test code:

```bash
bash scripts/verify-fast.sh
```

Equivalent commands:

```bash
cmake --preset gcc-release-tests
cmake --build --preset build-gcc-release-tests
ctest --preset test-gcc-release -j

uv sync --dev
uv pip install -e .
uv run python -m pytest -q -m "not benchmark"
```

Pass criteria:

- C++ unit tests pass on the primary toolchain.
- Editable package install succeeds from the repository root.
- Python tests pass without benchmark-only tests.

## Full Gate

Run this before handing off a substantial review batch or merging a risky change:

```bash
bash scripts/verify-full.sh
```

Prerequisite:

- `clang++` must be available in `PATH` for the Clang half of the gate.

Equivalent commands:

```bash
cmake --preset gcc-release-tests
cmake --build --preset build-gcc-release-tests
ctest --preset test-gcc-release -j

cmake --preset clang-release-tests
cmake --build --preset build-clang-release-tests
ctest --preset test-clang-release -j

uv sync --dev
uv pip install -e .
uv run python -m pytest -q -m "not benchmark"
```

Pass criteria:

- The same correctness surface passes under both GCC and Clang.
- The editable Python package still installs and imports after a clean rebuild.

## Release Gate

Run this before tagging a release:

```bash
bash scripts/verify-release.sh
```

Prerequisite:

- `clang++` must be available in `PATH`, because the release gate includes the `full` gate.

Equivalent commands:

```bash
bash scripts/verify-full.sh
bash scripts/package-smoke.sh
```

Optional release-only additions:

- `uv sync --dev --group bench`
- `uv run python -m pytest -q -m benchmark`
- `cmake --preset gcc-release -DTENSOR_ENABLE_BENCHMARKS=ON`
- `cmake --build --preset build-gcc-release`

Pass criteria:

- `full` gate passes.
- A wheel installs into a clean virtual environment.
- The installed package works from outside the repository tree.
- Manual consumer smoke checks pass.

## Package Smoke Workflow

The package smoke path is intentionally different from the normal dev-tree pytest path:

1. Build a wheel from the repository root.
2. Create a fresh virtual environment in a temporary directory.
3. Install the wheel into that environment.
4. Run the consumer example from outside the repo tree.

Use:

```bash
bash scripts/package-smoke.sh
```

That script runs `examples/python/package_smoke.py` after wheel installation and verifies:

- `import tensor`
- `import tensor_py`
- basic tensor creation through the packaged interface
- NumPy-visible shapes, dtypes, and values

## Deep Manual Checks

Use these when a release changes packaging, bindings, or the first supported tensor ops:

1. From a clean shell, install the wheel and run `examples/python/package_smoke.py` from a temporary directory.
2. Confirm that the example still works after deleting or moving the local `build/` tree.
3. Re-run `uv pip install -e .` after a native-code change and verify imports still resolve correctly.
4. If Python bindings changed, exercise the public API interactively with a short REPL session:

```bash
uv run python
>>> import tensor as T
>>> a = T.ones([2, 2], "f32")
>>> b = T.zeros([2, 2], "f32")
>>> a + 2.0 * b
```

## Feature Roadmap

Version lines should describe shipped capabilities, not verification machinery. Verification gates apply to every line below.

### 0.2.x Core Dense Foundation

Target outcome:

- `matmul` is a stable primitive with clear shape rules and deterministic behavior.
- Elementwise ops, reductions, and storage/view semantics are stable enough to support that dense path.
- Python bindings expose only the already-stable core.

Stable gate:

- `matmul` forward behavior is covered by unit tests and Python smoke checks.
- Invalid shapes fail predictably.
- The documented primary dense operations behave the same across the supported toolchains.

### 0.3.x Minimal Trainable Path

Target outcome:

- Reverse-mode autograd is stable for the first supported ops.
- `Linear`, one loss, and `SGD` form a dependable learnable path.
- Small training examples converge repeatably.

Stable gate:

- Gradient tests pass numerically and structurally.
- Tiny end-to-end learning smoke tests are green in both C++ and Python-facing verification where applicable.

### 0.4.x Python Package Usability

Target outcome:

- The installed package works as a normal dependency outside the repo tree.
- Examples reflect the stable C++ feature surface rather than experimental internals.
- Editable and wheel installs behave predictably.

Stable gate:

- Package smoke passes from a temporary directory.
- At least one consumer example stays green through the release gate.

### 0.5.x Model-Building Expansion

Target outcome:

- The library exposes the next carefully chosen building blocks after `Linear`.
- Added primitives clearly contribute to building small dense models.
- Scope remains intentionally narrow and does not drift into broad framework behavior.

Stable gate:

- Every added primitive has direct forward tests, gradient checks where relevant, and a documented role in a small composed model.
