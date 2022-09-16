import collections.abc
import json
import jsonschema
import numpy as np
import os
import parafields._parafields as _parafields

from matplotlib import pyplot as plt
from parafields.mpi import default_partitioning, MPI


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

    # Perform some additional validations not part of the schema
    assert len(config["grid"]["extensions"]) == len(config["grid"]["cells"])

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
    return RandomField(config, dtype=dtype, partitioning=partitioning)


# A mapping of numpy types to C++ type names
possible_types = {np.float64: "double", np.float32: "float"}

# Restriction of types that parafields was compiled with
available_types = {
    dt: t for dt, t in possible_types.items() if _parafields.has_precision(t)
}


class RandomField:
    def __init__(self, config, dtype=np.float64, partitioning=None):
        # Validate the given config
        self.config = validate_config(config)

        # Ensure that the given dtype is supported by parafields
        if dtype not in possible_types:
            raise NotImplementedError("Dtype not supported by parafields!")
        if dtype not in available_types:
            raise NotImplementedError(
                "parafields was not compiler for dtype, but could be!"
            )

        # Extract the seed from the configuration
        seed = self.config.get("seed", 0)

        # Extract the partitioning (function)
        if partitioning is None:
            partitioning = default_partitioning

        if MPI is None:
            partitioning = (1,) * len(self.config["grid"]["cells"])
        else:
            # If the given partitioning is a function, call it
            if isinstance(partitioning, collections.abc.Callable):
                partitioning = partitioning(
                    MPI.COMM_WORLD.size, self.config["grid"]["cells"]
                )

        # Instantiate a C++ class for the field generator
        dim = len(self.config["grid"]["extensions"])
        FieldType = getattr(_parafields, f"RandomField{dim}D_{available_types[dtype]}")
        self._field = FieldType(dict_to_parameter_tree(self.config), partitioning)

        # Trigger the generation process
        self._field.generate(seed)

        # Storage for lazy evaluation
        self._eval = None

    def evaluate(self):
        # Lazily evaluate the entire field
        if self._eval is None:
            self._eval = self._field.eval()
        return self._eval

    def _repr_png_(self):
        eval_ = self.evaluate()
        plt.imshow(eval_, interpolation="nearest")


def interactive_field_generation():
    """Interactively explore field generation in a Jupyter notebook"""

    # Check whether the extra requirements were installed
    try:
        import ipywidgets_jsonschema
    except ImportError:
        print("Please re-run pip installation with 'parafields[jupyter]'")
        return

    # Create widgets for the configuration
    schema = load_schema()
    form = ipywidgets_jsonschema.Form(schema)
    form.show()
