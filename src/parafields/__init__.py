# Export the version given in project metadata
from importlib import metadata

__version__ = metadata.version(__package__)
del metadata

from parafields.field import generate_field
from parafields.interactive import (
    interactive_add_trend_component,
    interactive_generate_field,
)
from parafields.mpi import MPI
