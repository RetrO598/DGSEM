#pragma once
#include <boundary_condition/boundary_condition_base.hpp>
#include <equations/equations.hpp>
#include <xtensor/core/xtensor_forward.hpp>

namespace DGSEM {
template <class Basis, class Equations>
struct PeriodicBoundary {
  using traits = equations::EquationTraits<Equations>;
  using value_type = typename traits::value_type;
  void operator()(const std::array<std::size_t, traits::NDIMS> n_cells,
                  const xt::xarray<value_type> &u,
                  xt::xarray<value_type> &surface_flux) {
    return;
  }
};

} // namespace DGSEM