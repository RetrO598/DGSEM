#pragma once

#include <Kokkos_Core.hpp>
#include <array>
#include <base/base.hpp>
#include <cstddef>
#include <equations/equations.hpp>
#include <impl/Kokkos_Profiling.hpp>
#include <traits/Kokkos_IterationPatternTrait.hpp>

namespace DGSEM {

/**
 * @brief Functor for interface flux calculation in Kokkos parallel loop.
 *
 * @tparam T The value type.
 * @tparam Equations The physical equations type.
 * @tparam Basis The polynomial basis type.
 * @tparam InterfaceHelper The interface helper type.
 * @tparam NDIMS The number of dimensions.
 */
template <class T, class Equations, class Basis, class InterfaceHelper,
          std::size_t NDIMS>
struct InterfaceFluxFunctor {
  using traits = equations::EquationTraits<Equations>;
  using DataArray = typename solution_type_traits<T, NDIMS>::DataArray;
  using IndexArray = typename index_type_traits<NDIMS>::IndexArray;

  InterfaceFluxFunctor(IndexArray left_neighbors_, const Equations &eq_,
                       DataArray u_, DataArray surface_flux_)
      : left_neighbors(left_neighbors_), eq(eq_), u(u_),
        surface_flux(surface_flux_) {}

  static void apply(IndexArray left_neighbors_, const Equations &eq_,
                    DataArray u_, DataArray surface_flux_,
                    std::array<std::size_t, NDIMS> n_elems_)
    requires(NDIMS == 1)
  {
    InterfaceFluxFunctor functor(left_neighbors_, eq_, u_, surface_flux_);
    Kokkos::parallel_for("interface_flux", n_elems_[0], functor);
  }

  static void apply(IndexArray left_neighbors_, const Equations &eq_,
                    DataArray u_, DataArray surface_flux_,
                    std::array<std::size_t, NDIMS> n_elems_)
    requires(NDIMS == 2)
  {
    InterfaceFluxFunctor functor(left_neighbors_, eq_, u_, surface_flux_);
    Kokkos::parallel_for("interface_flux",
                         Kokkos::MDRangePolicy<Kokkos::Rank<2>>(
                             {0, 0}, {n_elems_[0], n_elems_[1]}),
                         functor);
  }

  KOKKOS_INLINE_FUNCTION void operator()(const std::size_t &ielem) const
    requires(NDIMS == 1)
  {
    InterfaceHelper::interface_flux(left_neighbors, eq, ielem, u, surface_flux);
  }

  KOKKOS_INLINE_FUNCTION void operator()(const std::size_t &ielem,
                                         const std::size_t &jelem) const
    requires(NDIMS == 2)
  {
    InterfaceHelper::interface_flux(left_neighbors, eq, ielem, jelem, u,
                                    surface_flux);
  }

  IndexArray left_neighbors;
  const Equations eq;
  DataArray u;
  DataArray surface_flux;
};

} // namespace DGSEM
