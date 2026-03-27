import numpy as np
import pytest

import tensor as T


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
