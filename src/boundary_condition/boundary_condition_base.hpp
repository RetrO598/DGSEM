#pragma once
#include <array>
#include <boundary_condition/extrapolate_boundary.hpp>
#include <boundary_condition/periodic_boundary.hpp>
#include <cstddef>
#include <equations/equations.hpp>
#include <xtensor/core/xtensor_forward.hpp>
namespace DGSEM {
enum class BoundaryCondition {
  Periodic,
  Wall,
  Inflow,
  Outflow,
  Extrapolate,
  Custom,
};

template <class Basis, class Equations, class SurfaceFlux, std::size_t NDIMS,
          BoundaryCondition BoundaryType>
struct BoundaryImpl;

template <class Basis, class Equations, class SurfaceFlux, std::size_t NDIMS>
struct BoundaryImpl<Basis, Equations, SurfaceFlux, NDIMS,
                    BoundaryCondition::Periodic> {
  using traits = equations::EquationTraits<Equations>;
  using value_type = typename traits::value_type;

  inline static void apply(const std::array<std::size_t, NDIMS> &n_cells,
                           std::size_t iboundary,
                           const xt::xarray<value_type> &u,
                           xt::xarray<value_type> &surface_flux) {
    PeriodicBoundary<Basis, Equations> bc{};
    bc(n_cells, u, surface_flux);
  }
};

template <class Basis, class Equations, class SurfaceFlux, std::size_t NDIMS>
struct BoundaryImpl<Basis, Equations, SurfaceFlux, NDIMS,
                    BoundaryCondition::Extrapolate> {
  using traits = equations::EquationTraits<Equations>;
  using value_type = typename traits::value_type;

  inline static void apply(const Equations &eq,
                           const std::array<std::size_t, NDIMS> &n_cells,
                           std::size_t iboundary,
                           const xt::xarray<value_type> &u,
                           xt::xarray<value_type> &surface_flux) {
    ExtrapolateBoundary<Basis, Equations, SurfaceFlux> bc{};
    bc.template operator()<NDIMS>(eq, n_cells, iboundary, u, surface_flux);
  }
};

template <class Basis, class Equations, class SurfaceFlux, std::size_t NDIMS>
struct BoundaryDispatcher {
  using traits = equations::EquationTraits<Equations>;
  using value_type = typename traits::value_type;

  inline constexpr static void
  apply(const Equations &eq, BoundaryCondition bc,
        const std::array<std::size_t, NDIMS> &n_cells, std::size_t iboundary,
        const xt::xarray<value_type> &u, xt::xarray<value_type> &surface_flux) {

    switch (bc) {
    case BoundaryCondition::Periodic:
      BoundaryImpl<Basis, Equations, SurfaceFlux, NDIMS,
                   BoundaryCondition::Periodic>::apply(n_cells, iboundary, u,
                                                       surface_flux);
      break;
    case BoundaryCondition::Extrapolate:
      BoundaryImpl<Basis, Equations, SurfaceFlux, NDIMS,
                   BoundaryCondition::Extrapolate>::apply(eq, n_cells,
                                                          iboundary, u,
                                                          surface_flux);
      break;
    default:
      throw std::runtime_error("Unsupported BC");
    }
  }
};
} // namespace DGSEM
