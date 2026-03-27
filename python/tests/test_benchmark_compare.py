import numpy as np
import pytest

pytest.importorskip("pytest_benchmark")

try:
    import torch
except Exception:  # pragma: no cover
    torch = None

import tensor as T


@pytest.mark.benchmark(group="zeros")
@pytest.mark.parametrize("n", [64, 256, 1024])
@pytest.mark.parametrize("dtype", ["f32"])
def test_bench_zeros_ours(benchmark, n, dtype):
    shape = [n, n]

    def ours():
        T.zeros(shape, dtype)

    benchmark(ours)


@pytest.mark.benchmark(group="zeros")
@pytest.mark.parametrize("n", [64, 256, 1024])
def test_bench_zeros_numpy(benchmark, n):
    shape = [n, n]

    def numpy_zeros():
        np.zeros(shape, dtype=np.float32)

    benchmark(numpy_zeros)


@pytest.mark.skipif(torch is None, reason="torch not available")
@pytest.mark.benchmark(group="zeros")
@pytest.mark.parametrize("n", [64, 256, 1024])
def test_bench_zeros_torch(benchmark, n):
    shape = [n, n]

    def th():
        torch.zeros(shape, dtype=torch.float32)

    benchmark(th)


@pytest.mark.benchmark(group="ones")
@pytest.mark.parametrize("n", [64, 256, 1024])
@pytest.mark.parametrize("dtype", ["f32"])
def test_bench_ones_ours(benchmark, n, dtype):
    shape = [n, n]

    def ours():
        T.ones(shape, dtype)

    benchmark(ours)


@pytest.mark.benchmark(group="ones")
@pytest.mark.parametrize("n", [64, 256, 1024])
def test_bench_ones_numpy(benchmark, n):
    shape = [n, n]

    def numpy_ones():
        np.ones(shape, dtype=np.float32)

    benchmark(numpy_ones)


@pytest.mark.skipif(torch is None, reason="torch not available")
@pytest.mark.benchmark(group="ones")
@pytest.mark.parametrize("n", [64, 256, 1024])
def test_bench_ones_torch(benchmark, n):
    shape = [n, n]

    def th():
        torch.ones(shape, dtype=torch.float32)

    benchmark(th)
