#pragma once

#include <Kokkos_Macros.hpp>
#include <array>
#include <base/base.hpp>
#include <cstddef>
#include <equations/equations.hpp>
#include <space_integral/surface_integral.hpp>

namespace DGSEM {
template <class Basis, class Equations, class Element, std::size_t NDIMS>
struct SurfaceIntegralFunctor {
  using traits = equations::EquationTraits<Equations>;
  using T = typename traits::value_type;
  using DataArray = typename solution_type_traits<T, NDIMS>::DataArray;
  using BasisData = typename Basis::DeviceData;

  SurfaceIntegralFunctor(DataArray du_, DataArray surface_flux_,
                         const Equations& eq_)
      : du(du_), surface_flux(surface_flux_), eq(eq_),
        basis_data(Basis::device_data()) {}

  static void apply(DataArray u_, DataArray surface_flux_, const Equations& eq_,
                    std::array<std::size_t, NDIMS> n_elems_)
    requires(NDIMS == 1)
  {
    SurfaceIntegralFunctor functor(u_, surface_flux_, eq_);
    Kokkos::parallel_for("surface_integral", n_elems_[0], functor);
  }

  static void apply(DataArray u_, DataArray surface_flux_, const Equations& eq_,
                    std::array<std::size_t, NDIMS> n_elems_)
    requires(NDIMS == 2)
  {
    SurfaceIntegralFunctor functor(u_, surface_flux_, eq_);
    Kokkos::parallel_for("surface_integral",
                         Kokkos::MDRangePolicy<Kokkos::Rank<2>>(
                             {0, 0}, {n_elems_[0], n_elems_[1]}),
                         functor);
  }

  static void apply(DataArray u_, DataArray surface_flux_, const Equations& eq_,
                    std::array<std::size_t, NDIMS> n_elems_)
    requires(NDIMS == 3)
  {
    SurfaceIntegralFunctor functor(u_, surface_flux_, eq_);
    Kokkos::parallel_for("surface_integral",
                         Kokkos::MDRangePolicy<Kokkos::Rank<3>>(
                             {0, 0, 0},
                             {n_elems_[0], n_elems_[1], n_elems_[2]}),
                         functor);
  }

  KOKKOS_INLINE_FUNCTION void operator()(const std::size_t& ielem) const
    requires(NDIMS == 1)
  {
    SurfaceIntegral<Basis, Equations, Element>::integral(
        ielem, du, surface_flux, basis_data);
  }

  KOKKOS_INLINE_FUNCTION void operator()(const std::size_t& ielem,
                                         const std::size_t& jelem) const
    requires(NDIMS == 2)
  {
    SurfaceIntegral<Basis, Equations, Element>::integral(
        ielem, jelem, du, surface_flux, basis_data);
  }

  KOKKOS_INLINE_FUNCTION void operator()(const std::size_t& ielem,
                                         const std::size_t& jelem,
                                         const std::size_t& kelem) const
    requires(NDIMS == 3)
  {
    SurfaceIntegral<Basis, Equations, Element>::integral(
        ielem, jelem, kelem, du, surface_flux, basis_data);
  }

  const Equations eq;
  DataArray du;
  DataArray surface_flux;
  BasisData basis_data;
};

} // namespace DGSEM
