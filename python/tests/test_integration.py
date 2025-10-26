import numpy as np
import pytest

import tensor_py as T


@pytest.mark.integration
def test_integration_create_and_numpy_ops():
    a = T.ones([2, 3], "f32")
    b = T.zeros([2, 3], "f32")
    # numpy operations on returned arrays
    c = a + 2.0 * b
    assert c.shape == (2, 3)
    assert np.all(c == 1.0)


