#include <pybind11/numpy.h>
#include <pybind11/pybind11.h>
#include <pybind11/stl.h>

#include <dune/common/fvector.hh>

#include <parafields/randomfield.hh>

#include <memory>
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

#define GENERATE_FIELD_DIM(dim)                                                \
  using RandomField##dim##D =                                                  \
    Dune::RandomField::RandomField<GridTraits<double, double, dim>>;           \
  py::class_<RandomField##dim##D> field##dim##d(m, "RandomField" #dim "D");    \
  field##dim##d.def(py::init<Dune::ParameterTree>());                          \
  field##dim##d.def("generate",                                                \
                    [](RandomField##dim##D& self) { self.generate(); });       \
  field##dim##d.def(                                                           \
    "probe",                                                                   \
    [](const RandomField##dim##D& self, const py::array_t<double>& pos) {      \
      Dune::FieldVector<double, 1> out;                                        \
      Dune::FieldVector<double, dim> apos;                                     \
      std::copy(pos.data(), pos.data() + dim, apos.begin());                   \
      self.evaluate(apos, out);                                                \
      return out[0];                                                           \
    });                                                                        \
                                                                               \
  field##dim##d.def("eval", [](const RandomField##dim##D& self) {              \
    std::array<unsigned int, dim> size;                                        \
    std::vector<Dune::FieldVector<double, 1>> out;                             \
    self.bulkEvaluate(out, size);                                              \
    std::array<std::size_t, dim> strides;                                      \
    strides[0] = sizeof(double);                                               \
    for (std::size_t i = 1; i < dim; ++i)                                      \
      strides[i] = strides[i - 1] * size[i - 1];                               \
    return py::array(py::dtype("double"), size, strides, out[0].data());       \
  });

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

  // Expose the RandomField template instantiations
  GENERATE_FIELD_DIM(1)
  GENERATE_FIELD_DIM(2)
  GENERATE_FIELD_DIM(3)
}

} // namespace parafields
