import parafields._parafields as _parafields

import math


# Import the mpi4py MPI interface if we have it installed
try:
    from mpi4py import MPI
except ImportError:
    MPI = None

# Do a sanity check whether the user tries to run the pre-built,
# sequential version of parafields in a parallel context.
if MPI is not None and MPI.COMM_WORLD.size > 1 and _parafields.uses_fakempi():
    raise RuntimeError(
        "You are trying to run the sequential version of parafields in a parallel context. Please build parafields from source instead following the installation instructions."
    )


def _default_partitioning(i, size, P, dims, trydims, opt):
    if i > 0:
        for k in range(1, P + 1):
            if (P % k == 0) and (size[i] % k == 0):
                trydims[i] = k
                opt = _default_partitioning(i - 1, size, P // k, dims, trydims, opt)
    else:
        if size[0] % P == 0:
            trydims[0] = P

            m = -1.0
            for k in range(len(dims)):
                mm = size[k] / trydims[k]
                if math.fmod(size[k], trydims[k]) > 0.0001:
                    mm *= 3
                if mm > m:
                    m = mm

            if m < opt:
                opt = m
                dims[:] = trydims[:]

    return opt


def default_partitioning(P, cells):
    """The default load balancing mechanism for parafields"""
    dim = len(cells)
    opt = 1e100
    dims = [0] * dim
    trydims = [0] * dim
    _default_partitioning(dim - 1, cells, P, dims, trydims, opt)

    return tuple(dims)
