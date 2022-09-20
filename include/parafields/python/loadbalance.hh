#pragma once

#include <array>

namespace parafields {

/** @brief A no-op loadbalancer
 *
 * This class fulfills the interface expected by parafields, but takes
 * the output as input and correctly forwards it. We use this to compile
 * the bindings with exactly one load balancer class (otherwise we would
 * need compile time variation) and determine the actual loadbalancing
 * on the Python side.
 */
template<long unsigned int dim>
class FixedLoadBalance
{
public:
  FixedLoadBalance(std::array<int, dim> result)
    : result(result)
  {
  }

  void loadbalance(const std::array<int, dim>&,
                   int,
                   std::array<int, dim>& dims) const
  {
    dims = result;
  }

private:
  std::array<int, dim> result;
};

} // namespace parafields
