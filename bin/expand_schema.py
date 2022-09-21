#!/usr/bin/env python

# This script reads the file raw-schema.json and expands it into a
# more complex schema that applies an anyOf rule per dimension. This
# approach avoid code duplication within the schema file.

import copy
import json
import os

# Read the raw schema
with open(os.path.join(os.path.dirname(__file__), "raw-schema.json"), "r") as f:
    schema = json.load(f)

# Add copies per dimension
dims = [copy.deepcopy(schema), copy.deepcopy(schema), copy.deepcopy(schema)]

# Do the necessary manual adjustments
for i in range(3):
    dims[i]["title"] = f"Dimension: {i + 1}"
    dims[i]["properties"]["grid"]["properties"]["cells"]["minItems"] = i + 1
    dims[i]["properties"]["grid"]["properties"]["cells"]["maxItems"] = i + 1
    dims[i]["properties"]["grid"]["properties"]["cells"]["default"] = [
        dims[i]["properties"]["grid"]["properties"]["cells"]["default"][0]
    ] * (i + 1)
    dims[i]["properties"]["grid"]["properties"]["extensions"]["minItems"] = i + 1
    dims[i]["properties"]["grid"]["properties"]["extensions"]["maxItems"] = i + 1
    dims[i]["properties"]["grid"]["properties"]["extensions"]["default"] = [
        dims[i]["properties"]["grid"]["properties"]["extensions"]["default"][0]
    ] * (i + 1)

dims[0]["properties"]["fftw"]["properties"]["transposed"]["default"] = False
dims[2]["properties"]["stochastic"]["properties"]["covariance"]["enum"].remove("cubic")

# Create the expanded schema
expanded_schema = {"anyOf": [dims[0], dims[1], dims[2]]}

# And write it to file
with open(
    os.path.join(os.path.dirname(__file__), "..", "src", "parafields", "schema.json"),
    "w",
) as f:
    json.dump(expanded_schema, f)
