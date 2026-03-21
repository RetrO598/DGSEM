#pragma once
#include <array>
#include <cstddef>
#include <equations/equations.hpp>
#include <xtensor/core/xtensor_forward.hpp>

namespace DGSEM {
template <class Basis, class Equations, class SurfaceFlux>
struct ExtrapolateBoundary {
  using traits = equations::EquationTraits<Equations>;
  using value_type = typename traits::value_type;
  constexpr static std::size_t NVARS = traits::NVARS;

  template <std::size_t NDIMS>
  void operator()(const Equations &eq,
                  const std::array<std::size_t, NDIMS> n_cells,
                  std::size_t iboundary, const xt::xarray<value_type> &u,
                  xt::xarray<value_type> &surface_flux)
    requires(NDIMS == 1)
  {
    if (iboundary == 0) {
      std::array<value_type, NVARS> u_outer{};
      std::array<value_type, NVARS> u_boundary{};
      for (std::size_t var = 0; var < NVARS; ++var) {
        u_boundary[var] = u(0, 0, var);
        u_outer[var] = u(0, 0, var);
      }
      std::array<value_type, NVARS> boundary_flux =
          SurfaceFlux::numerical_flux(eq, u_outer, u_boundary);

      for (std::size_t var = 0; var < NVARS; ++var) {
        surface_flux(0, 0, var) = boundary_flux[var];
      }
    } else if (iboundary == 1) {
      std::array<value_type, NVARS> u_outer{};
      std::array<value_type, NVARS> u_boundary{};
      for (std::size_t var = 0; var < NVARS; ++var) {
        u_boundary[var] = u(n_cells[0] - 1, Basis::NNodes - 1, var);
        u_outer[var] = u(n_cells[0] - 1, Basis::NNodes - 1, var);
      }
      std::array<value_type, NVARS> boundary_flux =
          SurfaceFlux::numerical_flux(eq, u_outer, u_boundary);

      for (std::size_t var = 0; var < NVARS; ++var) {
        surface_flux(n_cells[0] - 1, 1, var) = boundary_flux[var];
      }
    }
  }
};
} // namespace DGSEM