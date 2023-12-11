# Welcome to parafields

[![License](https://img.shields.io/badge/License-BSD%203--Clause-orange.svg)](https://opensource.org/licenses/BSD-3-Clause)
[![GitHub Workflow Status](https://img.shields.io/github/actions/workflow/status/parafields/parafields/ci.yml?branch=main)](https://github.com/parafields/parafields/actions/workflows/ci.yml)
[![Documentation Status](https://readthedocs.org/projects/parafields/badge/)](https://parafields.readthedocs.io/)
[![codecov](https://codecov.io/gh/parafields/parafields/branch/main/graph/badge.svg)](https://codecov.io/gh/parafields/parafields)

`parafields` is a Python package that provides Gaussian random fields
based on circulant embedding. Core features are:

* Large variety of covariance functions: exponential, Gaussian, Matérn,
  spherical and cubic covariance functions, among others
* Generation of distributed fields using domain decomposition
  and MPI through `mpi4py`
* Uses `numpy` data structures to ease integration with the
  Python ecosystem of scientific software
* Optional caching of matrix-vector products
* Easy integration into e.g. [FEniCSx-based](https://fenicsproject.org) PDE solvers ([Example that is currently not tested as part of our CI](https://github.com/parafields/parafields/blob/main/jupyter/fenicsx.ipynb))

`parafields` implements these features through Python bindings to the [parafields-core C++ library](https://github.com/parafields/parafields-core).
The following options are supported in the backend but not yet in the Python bindings:

* axiparallel and full geometric anisotropy
* value transforms like log-normal, folded normal, or
  sign function (excursion set)
* Coarsening and refinement of random fields for multigrid/-scale methods

## Installation

`parafields` is available from PyPI and can be installed using `pip`:

```
python -m pip install parafields
```

This will install a sequential, pre-compiled version of `parafields`.
In order to use `parafields` in an MPI-parallel context, you need to
instead build the package from source:

```
python -m pip install --no-binary parafields -v parafields
```

This will build the package from source and link against your system MPI.

Additionally, `parafields` defines the following optional dependency sets:

* `jupyter`: All requirements for an interactive Jupyter interface to `parafields`
* `tests`: All requirements for running `parafields`'s unit tests
* `docs`: All requirements for buildings `parafields`'s Sphinx documentation

These optional dependencies can be installed by installing e.g. `parafields[jupyter]`.

## Usage

This is a minimal usage example of the `parafields` package:

![Minimum usage example](https://raw.githubusercontent.com/parafields/parafields/main/parafields.gif)

For more examples, check out the [parafields documentation](https://parafields.readthedocs.io/).

## Reporting Issues

If you need support with `parafields` or found a bug, consider a bug on
[the issue tracker](https://github.com/parafields/parafields/issues).

## Contributing

`parafields` welcomes external contributions. For the best possible contributor
experience, consider opening an issue on [the issue tracker](https://github.com/parafields/parafields/issues)
before you start developing. Announcing your intended development in this way allows us to clarify
whether it is in the scope of the package. Contributions are then done via a pull
request on the GitHub repository. Please also add your name to the list of copyright holders.

For a development installation of `parafields`, use the following instructions:

```bash
git clone https://github.com/parafields/parafields.git
cd parafields
python -m pip install -v --editable .[tests,docs,jupyter]
```

Before contributing, make sure that the unit tests pass and that new functionality is
covered by unit tests. The unit tests can be run using pytest:

```bash
# Sequential tests
python -m pytest

# Parallel tests
mpirun --oversubscribe -np 4 python -m pytest --only-mpi
```

In order to locally build the Sphinx documentation, use the following commands:

```bash
sphinx-build -t html ./doc ./html
```

## Acknowledgments

The [parafields-core C++ library](https://github.com/parafields/parafields-core) is
work by Ole Klein whichis supported by the federal ministry of education and research
of Germany (Bundesministerium für Bildung und Forschung) and the ministry of science, research
and arts of the federal state of Baden-Württemberg (Ministerium für Wissenschaft, Forschung und Kunst Baden-Württemberg).

The Python bindings are realized by the Scientific Software Center of Heidelberg University.
The Scientific Software Center is funded as part of the Excellence Strategy of the German Federal and State Governments.
