#pragma once

#include <mpi.h>
#include <mpi4py/mpi4py.h>
#include <pybind11/pybind11.h>

namespace parafields {

/** @brief A thing wrapper for the MPI communicator
 *
 * This wrapper class is required in order to create bindings for
 * the communicator class in pybind11. As the actual type of MPI_Comm
 * depends on the MPI implementation (OpenMPI: void*, MPICH: int)
 * directly using MPI_Comm in bindings would lead to a risk of
 * clashes with other bindings for that type.
 *
 * This class solves the problem by providing a type that can be used
 * in bindings, that implicitly casts to the actual communicator type.
 *
 * The concept is taken from https://stackoverflow.com/a/62449190
 */
struct MPI4PyCommunicator
{
  MPI4PyCommunicator() = default;
  MPI4PyCommunicator(MPI_Comm value)
    : value(value)
  {
  }
  operator MPI_Comm() { return value; }
  MPI_Comm value;
};

} // namespace parafields

namespace pybind11::detail {

/** The pybind11 type caster for MPI4PyCommunicator
 *
 * Taken from https://stackoverflow.com/a/62449190
 */
template<>
struct type_caster<parafields::MPI4PyCommunicator>
{
  PYBIND11_TYPE_CASTER(parafields::MPI4PyCommunicator, _("MPI4PyCommunicator"));

  bool load(handle src, bool)
  {
    PyObject* py_src = src.ptr();

    // Check that we have been passed an mpi4py communicator
    if (PyObject_TypeCheck(py_src, &PyMPIComm_Type)) {
      // Convert to regular MPI communicator
      value.value = *PyMPIComm_Get(py_src);
    } else {
      return false;
    }

    return !PyErr_Occurred();
  }

  // C++ -> Python
  static handle cast(parafields::MPI4PyCommunicator src,
                     return_value_policy /* policy */,
                     handle /* parent */)
  {
    // Create an mpi4py handle
    return PyMPIComm_New(src.value);
  }
};

} // namespace pybind11::detail
