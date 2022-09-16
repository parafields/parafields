import parafields._parafields as _parafields

# Import the mpi4py MPI interface if we have it installed
try:
    from mpi4py import MPI
except ImportError:
    MPI = None

# Do a sanity check whether the user tries to run the pre-built,
# sequential version of parafields in a parallel context.
if MPI is not None and _parafields.uses_fakempi():
    raise RuntimeError(
        "You are trying to run the sequential version of parafields in a parallel context. Please build parafields from source instead following the installation instructions."
    )
