# Welcome to parafields

[![License](https://img.shields.io/badge/License-BSD%203--Clause-orange.svg)](https://opensource.org/licenses/BSD-3-Clause)
[![GitHub Workflow Status](https://img.shields.io/github/workflow/status/parafields/parafields/CI)](https://github.com/parafields/parafields/actions?query=workflow%3ACI)
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
* Easy integration into [FEniCSx-based](https://fenicsproject.org) PDE solvers ([Example](https://github.com/parafields/parafields/blob/main/jupyter/fenicsx.ipynb))

`parafields` implements these features through Python bindings to the [parafields-core C++ library](https://github.com/parafields/parafields-core).
The following options are supported in the backend but not yet in the Python bindings:

* axiparallel and full geometric anisotropy
* value transforms like log-normal, folded normal, or
  sign function (excursion set)
* Coarsening and refinement of random fields for multigrid/-scale methods

## Usage

This is a minimal usage example of the `parafields` package:

![Minimum usage example](https://raw.githubusercontent.com/parafields/parafields/main/parafields.gif)

For more examples, check out the [parafields documentation](https://parafields.readthedocs.io/).

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

## Acknowledgments

The [parafields-core C++ library](https://github.com/parafields/parafields-core) is
work by Ole Klein whichis supported by the federal ministry of education and research
of Germany (Bundesministerium für Bildung und Forschung) and the ministry of science, research
and arts of the federal state of Baden-Württemberg (Ministerium für Wissenschaft, Forschung und Kunst Baden-Württemberg).

The Python bindings are realized by the Scientific Software Center of Heidelberg University.
The Scientific Software Center is funded as part of the Excellence Strategy of the German Federal and State Governments.
