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

    # Validate the given config
    schema = load_schema()
    jsonschema.validate(instance=config, schema=schema)
    return config


def generate_field(config):
    """Generate a random field"""
    config = validate_config(config)
    dim = len(config["grid"]["extensions"])
    FieldType = getattr(_parafields, f"RandomField{dim}D")
    return FieldType(dict_to_parameter_tree(config))


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
