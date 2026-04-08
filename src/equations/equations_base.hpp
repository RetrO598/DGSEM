#pragma once

#include "Kokkos_Macros.hpp"
#include <Kokkos_Core.hpp>
#include <cstddef>

namespace DGSEM {
namespace equations {
class Equations1DBase {

public:
  KOKKOS_INLINE_FUNCTION
  virtual ~Equations1DBase() = default;
};

class Equations2DBase {
public:
  KOKKOS_INLINE_FUNCTION
  virtual ~Equations2DBase() = default;
};

template <class Equations>
struct EquationTraits {
  using value_type = typename Equations::value_type;
  constexpr static std::size_t NDIMS = Equations::NDIMS;
  constexpr static std::size_t NVARS = Equations::NVARS;
};
} // namespace equations
} // namespace DGSEM
