from parafields.field import *

import pytest


@pytest.mark.parametrize("covariance", ("exponential", "gammaExponential"))
def test_generate(covariance):
    config = {
        "grid": {"extensions": (1, 1), "cells": (10, 10)},
        "stochastic": {"variance": 1, "covariance": covariance, "corrLength": 0.05},
    }
    field = generate_field(config)
