# If parafields is operating in parallel, we need to initialize
# MPI for the testing. If it is operating in sequential, this will
# throw an ImportError that we do not care about.
try:
    from mpi4py import MPI
except ImportError:
    pass
