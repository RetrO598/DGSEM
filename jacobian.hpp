#pragma once
#include "data_container.hpp"
#include "equations.hpp"
#include <cstddef>
#include <xtensor/core/xtensor_forward.hpp>
namespace DGSEM {
template <class Basis, class Equations, class Element>
struct JacobianProj;

template <class Basis, class Equations, class T>
struct JacobianProj<Basis, Equations, StructuredElementContainer<T, 1>> {
  using traits = equations::EquationTraits<Equations>;
  constexpr static std::size_t NVARS = traits::NVARS;

  inline constexpr static void
  apply(const StructuredElementContainer<T, 1> &element, std::size_t ielem,
        xt::xarray<T> &du) {
    for (std::size_t i = 0; i < Basis::NNodes; ++i) {
      T factor = -element.inverse_jacobian(ielem, i);
      for (std::size_t var = 0; var < NVARS; ++var) {
        du(ielem, i, var) *= factor;
      }
    }
  }
};
} // namespace DGSEM