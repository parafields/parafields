from parafields.field import *

import numpy as np
import pytest


@pytest.mark.parametrize(
    "covariance",
    (
        "exponential",
        "gammaExponential",
        "separableExponential",
        "matern32",
        "matern52",
        "gaussian",
        "spherical",
        "cauchy",
        "generalizedCauchy",
        "dampedOscillation",
        "whiteNoise",
    ),
)
@pytest.mark.parametrize("dim", (1, 2, 3))
@pytest.mark.parametrize("dtype", available_types.keys())
@pytest.mark.parametrize("embedding_type", ("classical",))
@pytest.mark.parametrize("seed", (0, 42))
@pytest.mark.parametrize("variance", (0.5, 1.0))
def test_generate(
    covariance,
    dim,
    dtype,
    embedding_type,
    seed,
    variance,
):
    field = generate_field(
        cells=[10] * dim,
        covariance=covariance,
        dtype=dtype,
        embedding_type=embedding_type,
        extensions=[1] * dim,
        seed=seed,
        variance=variance,
    )
    data = field.evaluate()
    for i in range(dim):
        assert data.shape[i] == 10

    # Check that the field can be reproduced
    field.seed = None
    field.generate(seed=seed)
    data2 = field.evaluate()

    assert np.allclose(data, data2)


def test_seed_is_none():
    field = generate_field(seed=None)
    arr1 = field.evaluate()

    # Generate a second time
    field.generate(seed=None)
    arr2 = field.evaluate()

    assert not np.allclose(arr1, arr2)


def test_sign_transform():
    field = generate_field(transform="sign")
    arr = field.evaluate()
    assert np.unique(arr).shape[0] == 2


def test_custom_transform():
    def trafo(a):
        return a + 1

    field = generate_field(transform=trafo)
    field.evaluate()
