Parallel Usage
==============

It is one of the unique features of parafields, that it allows you to generate Gaussian process in an MPI-parallel fashion.
This allows e.g. integration into MPI-parallel PDE solvers using domain decomposition methods.

Basic Usage
-----------

In order to allow parallel computations, parafields integrates with the mpi4py library which you should import before importing parafields:

.. code::

    from mpi4py import MPI
    import parafields

If you have not worked with mpi4py, it might be worthwhile to check their
`beginner tutorial <https://mpi4py.readthedocs.io/en/stable/tutorial.html#running-python-scripts-with-mpi>`_.
The minimum take away message is that in order to run your code in parallel,
you need to invoke the Python interpreter through `mpiexec` or your code
will run sequentially:

.. code::

    mpiexec -n 4 python yourscript.py

The parafields API described in the sequential usage section is unchanged,
only that the evaluation returns only the part relevant on the current rank:

.. code::

    field = parafields.generate_field(
        cells=(256, 256), extensions=(1.0, 1.0), covariance="exponential", variance=1.0
    )
    print(field.shape)

    // Run with 4 processors, this will print (128, 128) on each rank.


Passing communicators
---------------------

By default, parafields uses MPI.COMM_WORLD as the communicator.
If you want to use a different one, you can create it in mpi4py
and pass it to the generate_field function:

.. code::

    field = parafields.generate_field(comm=MPI.COMM_SELF)

Data Distribution
-----------------

The process of how parafields distributes the data to processors can be
customized by passing the partitioning argument to generate_field. You
can provide on of two things to partitioning:

* A tuple of integers whose length is the domain dimensions. The product of all entries must match the number of processors. E.g. to distribute to 8 processors using a 2x2x2 cube topology, you need to pass `(2, 2, 2)`.
* A function that accepts the number of processors and the resolution tuple as arguments and returns such tuple.

This is an example of such function that generates a striped topology:

.. code::

    def striped_partitioning(P, cells):
        result = [1] * len(cells)
        result[0] = P
        return tuple(result)

    field = parafields.generate_field(partitioning=striped_partitioning)

It should be noted that there are some constraints to the partitioning process
that arise from the internal workings of the FFTW library. 

* The number of cells in X direction should be divisible by the number of processors.
* As FFTW will internally always work with a striped partitioning.
  parafields still allows you to define arbitrary partitionings, but communication
  happens in order to get the data to the correct processor. Therefore, using a striped
  partitioning will always be the most computationally efficient partitioning
  (albeit not optimal for many domain decomposition applications).