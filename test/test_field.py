from parafields.field import *

from parafields.exceptions import NegativeEigenvalueError

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

    # As it is quite easy to have negative eigenvalues in the covariance
    # matrix, we ignore this error in systematic testing.
    try:
        data = field.evaluate()
    except RuntimeError as e:
        if e.args[0] != "negative eigenvalues in covariance matrix":
            raise e

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


def whitenoise(v, x):
    for i in x:
        if np.abs(i) > 1e-10:
            return 0.0
    return v


def exponential(v, x):
    return v * np.exp(-np.linalg.norm(x))


@pytest.mark.parametrize(
    "builtin,custom",
    [("whiteNoise", whitenoise), ("exponential", exponential)],
    ids=["whitenoise", "exponential"],
)
def test_custom_covariance(builtin, custom):
    field1 = generate_field(covariance=builtin, seed=0)
    field2 = generate_field(covariance=custom, seed=0)

    assert np.allclose(field1.evaluate(), field2.evaluate())


@pytest.mark.parametrize(
    "method",
    [
        "add_mean_trend_component",
        "add_slope_trend_component",
        "add_disk_trend_component",
        "add_block_trend_component",
    ],
)
def test_add_trend_component(method):
    field = generate_field()
    eval1 = field.evaluate()
    getattr(field, method)()
    eval2 = field.evaluate()

    assert not np.allclose(eval1, eval2)


def test_custom_rng():
    gen = np.random.default_rng()
    rng = lambda: gen.random()

    field = generate_field(rng=rng)
    field.evaluate()


def test_probe():
    # Generate field and bulk evaluate it
    field = generate_field(cells=(128, 128), extensions=(1.0, 1.0))
    eval_ = field.evaluate()

    # Iterate over all cells and do manual probing
    for i in range(128):
        for j in range(128):
            assert eval_[i, j] == field.probe(
                np.array([i * (1 / 128) + 1 / 256, j * (1 / 128) + 1 / 256])
            )


def test_negative_eigenvalues_throw_correct():
    with pytest.raises(NegativeEigenvalueError):
        generate_field(
            cells=(256, 256),
            extensions=(1.0, 1.0),
            covariance="gaussian",
            corrLength=0.5,
        )


def test_autotune_embedding_factor():
    # Generate a field that requires increased embedding factor
    field = generate_field(
        cells=(256, 256),
        extensions=(1.0, 1.0),
        covariance="gaussian",
        corrLength=0.5,
        autotune_embedding_factor=True,
    )

    assert field.embedding_factor == 6
