#pragma once

#include <containers/containers.hpp>
#include <cstddef>
#include <equations/equations.hpp>

namespace DGSEM {
template <class Basis, class Equations, class Element>
struct SurfaceIntegral;

template <class Basis, class Equations, class T>
struct SurfaceIntegral<Basis, Equations, StructuredElementContainer<T, 1>> {
  using traits = equations::EquationTraits<Equations>;
  constexpr static std::size_t NVARS = traits::NVARS;
  using BasisData = typename Basis::DeviceData;

  template <class ArrayU, class ArrayFlux>
  KOKKOS_INLINE_FUNCTION constexpr static void
  integral(std::size_t ielem, ArrayU &du, const ArrayFlux &surface_flux,
           const BasisData& basis_data) {
    T factor_1 = basis_data.boundary_interpolation_left[0];
    T factor_2 = basis_data.boundary_interpolation_right[Basis::NNodes - 1];
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
