from parafields.utils import *

import json
import jsonschema
import os
import pytest


def test_is_iterable():
    assert is_iterable([0, 1])
    assert is_iterable((0, 1))
    assert is_iterable(["abc", "def"])
    assert not is_iterable(42)
    assert not is_iterable("abc")


@pytest.mark.parametrize("schemafile", ["stochastic.json", "trend.json"])
def test_load_schema(schemafile):
    schema = load_schema(schemafile)
    assert isinstance(schema, dict)

    # Make sure that the given schema is a valid jsonschema
    filename = os.path.join(
        os.path.split(jsonschema.__file__)[0], "schemas", "draft7.json"
    )
    with open(filename, "r") as f:
        meta_schema = json.load(f)
    meta_schema["additionalProperties"] = False
    jsonschema.validate(instance=schema, schema=meta_schema)
