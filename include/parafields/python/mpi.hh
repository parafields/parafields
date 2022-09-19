#pragma once

#include <mpi.h>
#include <mpi4py/mpi4py.h>
#include <pybind11/pybind11.h>

namespace parafields {

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
