#pragma once

#include "base/global_def.hpp"
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

  // Legacy xtensor implementation
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

  // Kokkos implementation
  template <class Equations_, class SurfaceFlux_, class Mesh, class T,
            std::size_t NDIMS, class ArrayU, class ArrayFlux>
  KOKKOS_INLINE_FUNCTION void
  apply_device(const Mesh &mesh, const Equations &eq, const ArrayU &u,
               ArrayFlux &surface_flux, std::size_t face_id, T time,
               int index = 0) const {
    if constexpr (NDIMS == 1) {
      std::size_t n_cells = mesh.get_num_cells(0);
      std::size_t n_nodes = u.extent(1);

      if (face_id == 0) {
        std::array<T, NVARS> u_outer{};
        std::array<T, NVARS> u_boundary{};
        for (std::size_t var = 0; var < NVARS; ++var) {
          u_boundary[var] = u(0, 0, var);
          u_outer[var] = u(0, 0, var);
        }
        auto boundary_flux = SurfaceFlux::numerical_flux(eq, u_outer, u_boundary);
        for (std::size_t var = 0; var < NVARS; ++var) {
          surface_flux(0, 0, var) = boundary_flux[var];
        }
      } else {
        std::array<T, NVARS> u_outer{};
        std::array<T, NVARS> u_boundary{};
        for (std::size_t var = 0; var < NVARS; ++var) {
          u_boundary[var] = u(n_cells - 1, n_nodes - 1, var);
          u_outer[var] = u(n_cells - 1, n_nodes - 1, var);
        }
        auto boundary_flux = SurfaceFlux::numerical_flux(eq, u_boundary, u_outer);
        for (std::size_t var = 0; var < NVARS; ++var) {
          surface_flux(n_cells - 1, 1, var) = boundary_flux[var];
        }
      }
    }
  }
};
} // namespace DGSEM