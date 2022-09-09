import collections.abc
import json
import jsonschema
import os
import parafields._parafields as _parafields


def is_iterable(x):
    """Decide whether x is a non-string iterable"""
    if isinstance(x, str):
        return False
    return isinstance(x, collections.abc.Iterable)


def dict_to_parameter_tree(data, tree=_parafields.ParameterTree(), prefix=""):
    """Convert a (nested) dictionary to a C++ parameter tree structure"""
    for k, v in data.items():
        if isinstance(v, dict):
            dict_to_parameter_tree(v, tree=tree, prefix=prefix + k + ".")
        else:
            if is_iterable(v):
                v = " ".join([str(x) for x in v])
            tree.set(prefix + k, str(v))
    return tree


def validate_config(config):
    """Validate the given configuration against the provided schema"""
    # Load the schema file shipped with parafields
    schema_file = os.path.join(os.path.dirname(__file__), "schema.json")
    with open(schema_file, "r") as f:
        schema = json.load(f)

    # Validate the given config
    jsonschema.validate(instance=config, schema=schema)

    return config


def generate(config={}):
    return _parafields.RandomField2D(dict_to_parameter_tree(config))
