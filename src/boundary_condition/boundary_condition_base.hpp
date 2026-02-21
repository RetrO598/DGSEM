#pragma once
#include "../equations/equations.hpp"
#include <array>
#include <cstddef>
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

template <class Basis, class Equations, BoundaryCondition BoundaryType,
          std::size_t NDIMS>
struct BoundaryImpl;

template <class Basis, class Equations>
struct PeriodicBoundary {
  using traits = equations::EquationTraits<Equations>;
  using value_type = typename traits::value_type;
  void operator()(const xt::xarray<value_type> &u,
                  xt::xarray<value_type> &surface_flux) {
    return;
  }
};

template <class Basis, class Equations>
struct BoundaryImpl<Basis, Equations, BoundaryCondition::Periodic, 1> {
  using traits = equations::EquationTraits<Equations>;
  using value_type = typename traits::value_type;

  inline static void apply(const std::array<std::size_t, 1> &n_cells,
                           std::size_t iboundary,
                           const xt::xarray<value_type> &u,
                           xt::xarray<value_type> &surface_flux) {
    PeriodicBoundary<Basis, Equations> bc{};
    bc(u, surface_flux);
  }
};

template <class Basis, class Equations, std::size_t NDIMS>
struct BoundaryDispatcher {
  using traits = equations::EquationTraits<Equations>;
  using value_type = typename traits::value_type;

  inline constexpr static void
  apply(BoundaryCondition bc, const std::array<std::size_t, NDIMS> &n_cells,
        std::size_t iboundary, const xt::xarray<value_type> &u,
        xt::xarray<value_type> &surface_flux) {

    switch (bc) {
    case BoundaryCondition::Periodic:
      BoundaryImpl<Basis, Equations, BoundaryCondition::Periodic, NDIMS>::apply(
          n_cells, iboundary, u, surface_flux);
      break;

    default:
      throw std::runtime_error("Unsupported BC");
    }
  }
};
} // namespace DGSEM
