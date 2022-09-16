from parafields.mpi import *

import functools
import pytest


@pytest.mark.parametrize("p,cells", [(1, (1, 1)), (2, (2, 2)), (4, (4, 16))])
def test_default_loadbalancing(p, cells):
    partitioning = default_partitioning(p, cells)
    assert p == functools.reduce(lambda a, b: a * b, partitioning, 1)
