#pragma once
#include "data_container.hpp"
#include "equations.hpp"
#include <array>
#include <cstddef>
#include <xtensor/core/xtensor_forward.hpp>

namespace DGSEM {

template <class Basis, class Equations, class SurfaceFlux, class Element>
struct InterfaceHelper;

template <class Basis, class Equations, class SurfaceFlux, class T>
struct InterfaceHelper<Basis, Equations, SurfaceFlux,
                       StructuredElementContainer<T, 1>> {
  using traits = equations::EquationTraits<Equations>;
  constexpr static std::size_t NVARS = traits::NVARS;

  inline constexpr static void
  interface_flux(const StructuredElementContainer<T, 1> &element,
                 const Equations &eq, std::size_t ielem, xt::xarray<T> &u,
                 xt::xarray<T> &surface_flux) {
    std::size_t left_elem = element.left_neighbors(ielem, 0);
    if (left_elem != static_cast<std::size_t>(-1)) {
      std::array<T, NVARS> u_ll{};
      std::array<T, NVARS> u_rr{};
      for (std::size_t var = 0; var < NVARS; ++var) {
        u_ll[var] = u(left_elem, Basis::NNodes - 1, var);
        u_rr[var] = u(ielem, 0, var);
      }

      std::array<T, NVARS> interface =
          SurfaceFlux::numerical_flux(eq, u_ll, u_rr);

      for (std::size_t var = 0; var < NVARS; ++var) {
        surface_flux(left_elem, 1, var) = interface[var];
        surface_flux(ielem, 0, var) = interface[var];
      }
    }
  }
};

} // namespace DGSEM
