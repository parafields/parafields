from parafields.field import *

import pytest


@pytest.mark.parametrize("dim", (1, 2, 3))
@pytest.mark.parametrize("covariance", ("exponential", "gammaExponential"))
def test_generate(dim, covariance):
    config = {
        "grid": {"extensions": [1] * dim, "cells": [10] * dim},
        "stochastic": {"variance": 1, "covariance": covariance, "corrLength": [0.05]},
    }
    field = generate_field(config)
