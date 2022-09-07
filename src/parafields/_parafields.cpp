#include <pybind11/pybind11.h>

#include <parafields/randomfield.hh>

namespace py = pybind11;

namespace parafields {

PYBIND11_MODULE(_parafields, m)
{
  m.doc() = "The parafields Python package for (parallel) parameter field generation";
}

} // namespace parafields
