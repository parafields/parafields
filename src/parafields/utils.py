import collections.abc
import json
import os


def is_iterable(x):
    """Decide whether x is a non-string iterable"""
    if isinstance(x, str):
        return False
    return isinstance(x, collections.abc.Iterable)


def load_schema(filename):
    # Load the schema file shipped with parafields
    schema_file = os.path.join(os.path.dirname(__file__), filename)
    with open(schema_file, "r") as f:
        return json.load(f)
