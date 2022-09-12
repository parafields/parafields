from parafields.field import *

import pytest


@pytest.mark.parametrize("dim", (1, 2, 3))
@pytest.mark.parametrize("covariance", ("exponential", "gammaExponential"))
def test_generate(dim, covariance):
    field = generate_field(
        cells=[10] * dim, extensions=[1] * dim, covariance="exponential"
    )
    data = field.evaluate()
    for i in range(dim):
        assert data.shape[i] == 10
