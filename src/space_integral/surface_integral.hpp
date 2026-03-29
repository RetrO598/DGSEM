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

template <class Basis, class Equations, class T>
struct SurfaceIntegral<Basis, Equations, StructuredElementContainer<T, 2>> {
  using traits = equations::EquationTraits<Equations>;
  constexpr static std::size_t NVARS = traits::NVARS;
  using BasisData = typename Basis::DeviceData;

  KOKKOS_INLINE_FUNCTION static std::size_t local_dof(std::size_t inode,
                                                      std::size_t jnode) {
    return jnode * Basis::NNodes + inode;
  }

  KOKKOS_INLINE_FUNCTION static std::size_t face_dof(std::size_t face,
                                                     std::size_t node) {
    return face * Basis::NNodes + node;
  }

  template <class ArrayU, class ArrayFlux>
  KOKKOS_INLINE_FUNCTION constexpr static void
  integral(std::size_t ielem, std::size_t jelem, ArrayU &du,
           const ArrayFlux &surface_flux, const BasisData& basis_data) {
    const T factor_1 = basis_data.boundary_interpolation_left[0];
    const T factor_2 =
        basis_data.boundary_interpolation_right[Basis::NNodes - 1];

    for (std::size_t jnode = 0; jnode < Basis::NNodes; ++jnode) {
      const std::size_t left_dof = local_dof(0, jnode);
      const std::size_t right_dof = local_dof(Basis::NNodes - 1, jnode);
      for (std::size_t var = 0; var < NVARS; ++var) {
        du(ielem, jelem, left_dof, var) -=
            surface_flux(ielem, jelem, face_dof(0, jnode), var) * factor_1;
        du(ielem, jelem, right_dof, var) +=
            surface_flux(ielem, jelem, face_dof(1, jnode), var) * factor_2;
      }
    }

    for (std::size_t inode = 0; inode < Basis::NNodes; ++inode) {
      const std::size_t bottom_dof = local_dof(inode, 0);
      const std::size_t top_dof = local_dof(inode, Basis::NNodes - 1);
      for (std::size_t var = 0; var < NVARS; ++var) {
        du(ielem, jelem, bottom_dof, var) -=
            surface_flux(ielem, jelem, face_dof(2, inode), var) * factor_1;
        du(ielem, jelem, top_dof, var) +=
            surface_flux(ielem, jelem, face_dof(3, inode), var) * factor_2;
      }
    }
  }
};
} // namespace DGSEM
