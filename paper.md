---
title: ""
tags:
  - Python
  - MPI
  - scientific computing
  - high performance computing
  - uncertainty quantification
  - random field generation
  - circulant embedding
authors:
  - name: Dominic Kempf
    orcid: 0000-0002-6140-2332
    affiliation: "1, 2"
    corresponding: true
    equal-contrib: true
  - name: Ole Klein
    orcid: 0000-0002-3295-7347
    equal-contrib: true
    affiliation: 4
  - name: Robert Kutri
    affiliation: "2, 3"
  - name: Robert Scheichl
    affiliation: "2, 3"
    orcid: 0000-0001-8493-4393
  - name: Peter Bastian
    affiliation: 2
affiliations:
 - name: Scientific Software Center, Heidelberg University
   index: 1
 - name: Interdisciplinary Center for Scientific Computing, Heidelberg University
   index: 2
 - name: Institute for Mathematics, Heidelberg University
   index: 3
 - name: Independent Researcher
   index: 4
date: 11 May 2023
bibliography: paper.bib
---

# Summary

Parafields is a Python package for the generation of spatially heterogeneous parameter fields
with well-defined, known statistical properties. The use of such fields is a key ingredient of simulation
workflows that involve uncertain, spatially heterogeneous parameters. Application areas include
e.g. soil physics and material sciences.

Parafields is also able to run in parallel using the Message Passing Interface (MPI) standard through mpi4py [@mpi4py].
In this case, the computational domain is split and only the part of the random field relevant
to a certain process is generated on that process. The generation process is implemented in a performance-oriented
C++ backend library and exposed in Python though an intuitive Python interface.

# Statement of need

*This should summarize why we need this piece of research software, including naming competitors*

Circulant embedding references: [@dietrich1997fast], [@graham2018analysis], [@chan1999simulation].
Original work: [@KLEIN2017161]

# Implementation

Parafields looks back at over ten years of development history: It was first implemented as an extension to the
Dune framework [@dune] for the numerical solution of partial differential equations. This is restriced the potential
user base to users of that software framework, although there was quite some interest in the software from outside this community.
In 2022, we started a huge refactoring: The previous C++ code base [@dune-randomfield] got rewritten to have a weaker dependency on Dune, which
e.g. included a rewrite of the CMake build system [@parafields-core]. In order to open up to a wider user base, a Python interface based written in pybind11 [@pybind11] was added.

When engineering the Python package, we put special emphasis on the following usability aspects: Installability, customizabilility and embedding into existing user workflows.

The recommended installation procedure for parafields is perfectly aligned with the state-of-the-art of the Python language: It is installable through with `pip` and automatically compiles using the CMake build systen of the project through scikit-build [@scikit-build]. Required dependency of the C++ library are automatically fetched and built in the required configuration. For sequential use, we also provide
pre-compiled Python wheels. They are built against the sequential MPI stub library FakeMPI [@fakempi], which allows us to build the sequential and the parallel version from the same code base. Users that want to leverage MPI through mpi4py will instead build the package from source against their system MPI library.

It was a goal of the design of the Python API to expose as much of the flexibility of the underlying C++ framework as possible.
In order to do so, we use pybind11's capabilities to pass Python callables to the C++ backend.
This allows users to e.g. implement custom covariance functions or random number generators.

In order to embed perfectly into existing user workflows, the Python API is entirely based on numpy arrays [@numpy].
Furthermore, we acknowledge the fact that many Python users write scientific applications within Jupyter: Our fields render nicely as images in Jupyter and field generation can optionally be configured
through an interactive widget frontend within Jupyter.

# Applications

*This could either be a single research application using this or a description of the problem class being solved with this*

# Acknowledgments

The authors thank all contributors of the dune-randomfield project for their valueable contributions that are now part of the parafields-core library.
Dominic Kempf is employed by the Scientific Software Center of Heidelberg University which is funded as part of the Excellence Strategy of the German Federal and State Governments.
Ole Klein's work is supported by the federal ministry of education and research
of Germany (Bundesministerium für Bildung und Forschung) and the ministry of science, research
and arts of the federal state of Baden-Württemberg (Ministerium für Wissenschaft, Forschung und Kunst Baden-Württemberg).
