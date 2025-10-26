import numpy as np
import pytest

try:
    import torch  # optional
except Exception:  # pragma: no cover
    torch = None

import tensor_py as T


@pytest.mark.parametrize("shape,dtype", [([2, 3], "f32"), ([4], "i32")])
def test_zeros_ones_parity(shape, dtype):
    a = T.zeros(shape, dtype)
    b = T.ones(shape, dtype)
    if dtype in ("f32", "f64"):
        assert np.all(a == 0)
        assert np.all(b == 1)
    else:
        assert np.all(a == 0)
        assert np.all(b == 1)


@pytest.mark.benchmark(group="zeros")
@pytest.mark.parametrize("n", [64, 256, 1024])
@pytest.mark.parametrize("dtype", ["f32"])  # extend as supported
def test_bench_zeros_numpy(benchmark, n, dtype):
    shape = [n, n]
    # our
    def ours():
        T.zeros(shape, dtype)

    res = benchmark(ours)
    # numpy
    numpy_t = np.zeros(shape, dtype=np.float32)
    assert numpy_t.shape == tuple(shape)


@pytest.mark.skipif(torch is None, reason="torch not available")
@pytest.mark.benchmark(group="zeros")
@pytest.mark.parametrize("n", [64, 256, 1024])
@pytest.mark.parametrize("dtype", ["f32"])  # extend as supported
def test_bench_zeros_torch(benchmark, n, dtype):
    shape = [n, n]

    def th():
        torch.zeros(shape, dtype=torch.float32)

    benchmark(th)
