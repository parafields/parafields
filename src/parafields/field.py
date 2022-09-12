import collections.abc
import json
import jsonschema
import numpy as np
import os
import parafields._parafields as _parafields

from matplotlib import pyplot as plt


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
    }

    # Return the Python class representing the field
    return RandomField(config, dtype=dtype)


class RandomField:
    def __init__(self, config, dtype=np.float64):
        # Validate the given config
        self.config = validate_config(config)

        # We currently only support double precision
        assert dtype == np.float64

        # Instantiate a C++ class for the field generator
        dim = len(self.config["grid"]["extensions"])
        FieldType = getattr(_parafields, f"RandomField{dim}D")
        self._field = FieldType(dict_to_parameter_tree(self.config))

        # Trigger the generation process
        self._field.generate()

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
