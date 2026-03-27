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

  SurfaceIntegralFunctor(DataArray du_, DataArray surface_flux_,
                         const Equations& eq_)
      : du(du_), surface_flux(surface_flux_), eq(eq_) {}

  static void apply(DataArray u_, DataArray surface_flux_, const Equations& eq_,
                    std::array<std::size_t, NDIMS> n_elems_)
    requires(NDIMS == 1)
  {
    SurfaceIntegralFunctor functor(u_, surface_flux_, eq_);
    Kokkos::parallel_for("surface_integral", n_elems_[0], functor);
  }

  KOKKOS_INLINE_FUNCTION void operator()(const std::size_t& ielem) const
    requires(NDIMS == 1)
  {
    SurfaceIntegral<Basis, Equations, Element>::integral(ielem, du,
                                                         surface_flux);
  }

  const Equations eq;
  DataArray du;
  DataArray surface_flux;
};

}  // namespace DGSEM