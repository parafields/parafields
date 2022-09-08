#include <pybind11/pybind11.h>

#include <dune/common/fvector.hh>

#include <parafields/python.hpp>
#include <parafields/randomfield.hh>

#include <string>

namespace py = pybind11;

namespace parafields {

template<typename DF, typename RF, unsigned int dimension>
class GridTraits
{
public:
  enum
  {
    dim = dimension
  };

  using RangeField = RF;
  using Scalar = Dune::FieldVector<RF, 1>;
  using DomainField = DF;
  using Domain = Dune::FieldVector<DF, dim>;
};

PYBIND11_MODULE(_parafields, m)
{
  m.doc() =
    "The parafields Python package for (parallel) parameter field generation";

  // Expose the Dune::ParameterTree class to allow the Python side to
  // easily feed data into the C++ code.
  py::class_<Dune::ParameterTree> param(m, "ParameterTree");
  param.def(py::init<>());
  param.def("set",
            [](Dune::ParameterTree& self, std::string key, std::string value) {
              self[key] = value;
            });

  // Expose the 2D RandomField class
  using RandomField2D =
    Dune::RandomField::RandomField<GridTraits<double, double, 2>>;
  py::class_<RandomField2D> field2d(m, "RandomField2D");
  field2d.def(py::init<Dune::ParameterTree>());
}

} // namespace parafields
