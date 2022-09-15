from parafields.field import *

import numpy as np
import pytest


@pytest.mark.parametrize("dim", (1, 2, 3))
@pytest.mark.parametrize("dtype", available_types.keys())
@pytest.mark.parametrize("seed", (0, 42))
def test_generate(dim, dtype, seed):
    field = generate_field(
        cells=[10] * dim,
        extensions=[1] * dim,
        covariance="exponential",
        seed=seed,
        dtype=dtype,
    )
    data = field.evaluate()
    for i in range(dim):
        assert data.shape[i] == 10
