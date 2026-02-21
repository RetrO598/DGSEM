#pragma once

#include "../containers/data_container.hpp"
#include <cstddef>
#include <equations/equations.hpp>
#include <xtensor/core/xtensor_forward.hpp>
namespace DGSEM {
template <class Basis, class Equations, class Element>
struct SurfaceIntegral;

template <class Basis, class Equations, class T>
struct SurfaceIntegral<Basis, Equations, StructuredElementContainer<T, 1>> {
  using traits = equations::EquationTraits<Equations>;
  constexpr static std::size_t NVARS = traits::NVARS;

  inline constexpr static void
  integral(const StructuredElementContainer<T, 1> &element, std::size_t ielem,
           xt::xarray<T> &du, xt::xarray<T> &surface_flux) {
    T factor_1 = Basis::boundary_interpolation_left[0];
    T factor_2 = Basis::boundary_interpolation_right[Basis::NNodes - 1];
    for (std::size_t var = 0; var < NVARS; ++var) {
      du(ielem, 0, var) =
          (du(ielem, 0, var) - surface_flux(ielem, 0, var) * factor_1);

      du(ielem, Basis::NNodes - 1, var) =
          (du(ielem, Basis::NNodes - 1, var) +
           surface_flux(ielem, 1, var) * factor_2);
    }
  }
};
} // namespace DGSEM