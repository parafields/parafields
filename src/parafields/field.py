import collections.abc
import json
import jsonschema
import numpy as np
import os
import parafields._parafields as _parafields
import time

from matplotlib import cm
from parafields.mpi import default_partitioning, MPI
from PIL import Image


def is_iterable(x):
    """Decide whether x is a non-string iterable"""
    if isinstance(x, str):
        return False
    return isinstance(x, collections.abc.Iterable)


def dict_to_parameter_tree(data, tree=None, prefix=""):
    """Convert a (nested) dictionary to a C++ parameter tree structure"""
    if tree is None:
        tree = _parafields.ParameterTree()
    for k, v in data.items():
        if isinstance(v, dict):
            dict_to_parameter_tree(v, tree=tree, prefix=prefix + k + ".")
        else:
            if is_iterable(v):
                v = " ".join([str(x) for x in v])
            tree.set(prefix + k, str(v))
    return tree


def load_schema():
    # Load the schema file shipped with parafields
    schema_file = os.path.join(os.path.dirname(__file__), "schema.json")
    with open(schema_file, "r") as f:
        return json.load(f)


def validate_config(config):
    """Validate the given configuration against the provided schema"""

    # Validate the given config against the schema
    schema = load_schema()
    jsonschema.validate(instance=config, schema=schema)

    return config


def generate_field(
    cells=(512, 512),
    extensions=(1.0, 1.0),
    covariance="exponential",
    variance=1.0,
    corrLength=0.05,
    dtype=np.float64,
    seed=0,
    partitioning=None,
    comm=None,
):
    # The backend expects corrLength as a list
    if not is_iterable(corrLength):
        corrLength = [corrLength]

    # Create the backend configuration
    config = {
        "grid": {"cells": list(cells), "extensions": list(extensions)},
        "stochastic": {
            "corrLength": corrLength,
            "covariance": covariance,
            "variance": variance,
        },
        "seed": seed,
    }

    # Return the Python class representing the field
    return RandomField(
        config, dtype=dtype, partitioning=partitioning, comm=comm, seed=seed
    )


# A mapping of numpy types to C++ type names
possible_types = {np.float64: "double", np.float32: "float"}

# Restriction of types that parafields was compiled with
available_types = {
    dt: t for dt, t in possible_types.items() if _parafields.has_precision(t)
}


class RandomField:
    def __init__(
        self, config, dtype=np.float64, partitioning=None, comm=None, seed=None
    ):
        # Validate the given config
        self.config = validate_config(config)
        self.seed = None

        # Ensure that the given dtype is supported by parafields
        if dtype not in possible_types:
            raise NotImplementedError("Dtype not supported by parafields!")
        if dtype not in available_types:
            raise NotImplementedError(
                "parafields was not compiler for dtype, but could be!"
            )

        # Extract the partitioning (function)
        if partitioning is None:
            partitioning = default_partitioning

        if MPI is not None:
            # We use COMM_WORLD as the default communicator if running in parallel
            if comm is None:
                comm = MPI.COMM_WORLD
            # If the given partitioning is a function, call it
            if isinstance(partitioning, collections.abc.Callable):
                partitioning = partitioning(
                    MPI.COMM_WORLD.size, self.config["grid"]["cells"]
                )

        # Instantiate a C++ class for the field generator
        dim = len(self.config["grid"]["extensions"])
        FieldType = getattr(_parafields, f"RandomField{dim}D_{available_types[dtype]}")

        if comm is None:
            self._field = FieldType(dict_to_parameter_tree(self.config))
        else:
            self._field = FieldType(
                dict_to_parameter_tree(self.config), partitioning, comm
            )

        # Trigger the generation process
        self.generate(seed=seed)

        # Storage for lazy evaluation
        self._eval = None

    def generate(self, seed=None):
        """Regenerate the field with the given seed

        :param seed:
            The seed to use. If seed is None, a new seed will be used
            on every call.
        :type seed: int
        """

        # Maybe create a new seed
        if seed is None:
            seed = time.clock_gettime_ns(0) % (2**32)

        # Maybe invalidate any evaluations we have cached
        if seed != self.seed:
            self._eval = None
            self.seed = seed

            # Trigger field generation in the backend
            self._field.generate(seed)

    def evaluate(self):
        # Lazily evaluate the entire field
        if self._eval is None:
            self._eval = self._field.eval()
        return self._eval

    def _repr_png_(self):
        # Evaluate the field
        eval_ = self.evaluate()

        # If this is not 2D, we skip visualization
        if len(eval_.shape) != 2:
            return

        # Convert to PIL array
        return Image.fromarray(np.uint8(cm.gist_earth(eval_) * 255))
