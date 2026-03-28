"""Consumer-facing package smoke example for installed Tensor builds."""

from __future__ import annotations

import importlib

import numpy as np

import tensor as T


def main() -> None:
    ones = T.ones([2, 3], "f32")
    zeros = T.zeros([2, 3], "f32")

    assert ones.shape == (2, 3)
    assert zeros.shape == (2, 3)
    assert ones.dtype == np.float32
    assert zeros.dtype == np.float32
    assert np.all(ones == 1.0)
    assert np.all(zeros == 0.0)

    mixed = ones + (2.0 * zeros)
    assert np.allclose(mixed, np.ones((2, 3), dtype=np.float32))

    tensor_py = importlib.import_module("tensor_py")
    raw = tensor_py.ones([1, 4], "i32")
    assert raw.shape == (1, 4)
    assert raw.dtype == np.int32
    assert np.all(raw == 1)

    print("Tensor package smoke check passed.")


if __name__ == "__main__":
    main()
