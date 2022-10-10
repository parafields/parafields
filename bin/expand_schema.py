#!/usr/bin/env python

# This script reads the files raw-*.json and expands them into a
# more complex schemas that applies an anyOf rule per dimension. This
# approach avoids code duplication within the schema files.

import copy
import itertools
import json
import os

# Read the raw schemas
with open(os.path.join(os.path.dirname(__file__), "raw-stochastic.json"), "r") as f:
    schema = json.load(f)

with open(os.path.join(os.path.dirname(__file__), "raw-trend.json"), "r") as f:
    trend_schema = json.load(f)

# Add copies per dimension
dims = [copy.deepcopy(schema), copy.deepcopy(schema), copy.deepcopy(schema)]
trend_dims = [
    copy.deepcopy(trend_schema),
    copy.deepcopy(trend_schema),
    copy.deepcopy(trend_schema),
]

# Do the necessary manual adjustments to stochastic schema
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

    # Create sizes and defaults for axiparallel correlation length
    dims[i]["properties"]["stochastic"]["else"]["else"]["else"]["else"]["then"][
        "properties"
    ]["corrLength"]["minItems"] = (i + 1)
    dims[i]["properties"]["stochastic"]["else"]["else"]["else"]["else"]["then"][
        "properties"
    ]["corrLength"]["maxItems"] = (i + 1)
    axicorr = [0.05]
    for j in range(i):
        axicorr.append(axicorr[-1] * 2)
    dims[i]["properties"]["stochastic"]["else"]["else"]["else"]["else"]["then"][
        "properties"
    ]["corrLength"]["default"] = axicorr

    # Create sizes and defaults for geometric correlation length
    dims[i]["properties"]["stochastic"]["else"]["else"]["else"]["else"]["else"]["then"][
        "properties"
    ]["corrLength"]["minItems"] = (i + 1) * (i + 1)
    dims[i]["properties"]["stochastic"]["else"]["else"]["else"]["else"]["else"]["then"][
        "properties"
    ]["corrLength"]["maxItems"] = (i + 1) * (i + 1)
    geocorr = []
    for j, k in itertools.product(range(i + 1), range(i + 1)):
        if j == k:
            geocorr.append(0.05)
        else:
            geocorr.append(0.0)
    dims[i]["properties"]["stochastic"]["else"]["else"]["else"]["else"]["else"]["then"][
        "properties"
    ]["corrLength"]["default"] = geocorr
    del dims[i]["properties"]["stochastic"]["else"]["else"]["else"]["else"]["else"][
        "then"
    ]["properties"]["corrLength"]["items"]["exclusiveMinimum"]

# Apply some constraints and exceptions
dims[0]["properties"]["fftw"]["properties"]["transposed"]["default"] = False
dims[2]["properties"]["stochastic"]["properties"]["covariance"]["enum"].remove("cubic")

# Do the necessary manual adjustments to stochastic schema
for i in range(3):
    trend_dims[i]["title"] = f"Dimension: {i + 1}"
    trend_dims[i]["anyOf"][1]["properties"]["slope"]["properties"]["mean"][
        "maxItems"
    ] = (i + 1)
    trend_dims[i]["anyOf"][1]["properties"]["slope"]["properties"]["mean"][
        "minItems"
    ] = (i + 1)
    trend_dims[i]["anyOf"][1]["properties"]["slope"]["properties"]["variance"][
        "maxItems"
    ] = (i + 1)
    trend_dims[i]["anyOf"][1]["properties"]["slope"]["properties"]["variance"][
        "minItems"
    ] = (i + 1)
    trend_dims[i]["anyOf"][2]["properties"]["disk0"]["properties"]["mean_position"][
        "maxItems"
    ] = (i + 1)
    trend_dims[i]["anyOf"][2]["properties"]["disk0"]["properties"]["mean_position"][
        "minItems"
    ] = (i + 1)
    trend_dims[i]["anyOf"][2]["properties"]["disk0"]["properties"]["variance_position"][
        "maxItems"
    ] = (i + 1)
    trend_dims[i]["anyOf"][2]["properties"]["disk0"]["properties"]["variance_position"][
        "minItems"
    ] = (i + 1)
    trend_dims[i]["anyOf"][3]["properties"]["block0"]["properties"]["mean_position"][
        "maxItems"
    ] = (i + 1)
    trend_dims[i]["anyOf"][3]["properties"]["block0"]["properties"]["mean_position"][
        "minItems"
    ] = (i + 1)
    trend_dims[i]["anyOf"][3]["properties"]["block0"]["properties"]["mean_extent"][
        "maxItems"
    ] = (i + 1)
    trend_dims[i]["anyOf"][3]["properties"]["block0"]["properties"]["mean_extent"][
        "minItems"
    ] = (i + 1)
    trend_dims[i]["anyOf"][3]["properties"]["block0"]["properties"][
        "variance_position"
    ]["maxItems"] = (i + 1)
    trend_dims[i]["anyOf"][3]["properties"]["block0"]["properties"][
        "variance_position"
    ]["minItems"] = (i + 1)
    trend_dims[i]["anyOf"][3]["properties"]["block0"]["properties"]["variance_extent"][
        "maxItems"
    ] = (i + 1)
    trend_dims[i]["anyOf"][3]["properties"]["block0"]["properties"]["variance_extent"][
        "minItems"
    ] = (i + 1)

# Create the expanded schema
expanded_schema = {"anyOf": [dims[0], dims[1], dims[2]]}
trend_expanded_schema = {"anyOf": [trend_dims[0], trend_dims[1], trend_dims[2]]}

# And write them to file
with open(
    os.path.join(
        os.path.dirname(__file__), "..", "src", "parafields", "stochastic.json"
    ),
    "w",
) as f:
    json.dump(expanded_schema, f)

with open(
    os.path.join(os.path.dirname(__file__), "..", "src", "parafields", "trend.json"),
    "w",
) as f:
    json.dump(trend_expanded_schema, f)
