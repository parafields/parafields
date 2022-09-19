from parafields.mpi import *

from parafields.field import generate_field

import functools
import pytest

try:
    from mpi4py import MPI

    comms = [MPI.COMM_WORLD, MPI.COMM_SELF]
except ImportError:
    comms = []


@pytest.mark.parametrize("p,cells", [(1, (1, 1)), (2, (2, 2)), (4, (4, 16))])
def test_default_loadbalancing(p, cells):
    partitioning = default_partitioning(p, cells)
    assert p == functools.reduce(lambda a, b: a * b, partitioning, 1)


@pytest.mark.mpi
@pytest.mark.parametrize("comm", comms)
def test_generate_with_communicator(comm):
    field = generate_field(comm=comm)
