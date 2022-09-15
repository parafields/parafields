from parafields.field import *

import pytest


@pytest.mark.parametrize("dim", (1, 2, 3))
@pytest.mark.parametrize("seed", (0, 42))
def test_generate(dim, seed):
    field = generate_field(
        cells=[10] * dim, extensions=[1] * dim, covariance="exponential", seed=seed
    )
    data = field.evaluate()
    for i in range(dim):
        assert data.shape[i] == 10
