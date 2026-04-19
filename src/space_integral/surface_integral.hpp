#pragma once

#include <containers/containers.hpp>
#include <cstddef>
#include <equations/equations.hpp>
#include <utils/local_dof.hpp>

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
  integral(std::size_t ielem, ArrayU& du, const ArrayFlux& surface_flux,
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

  KOKKOS_INLINE_FUNCTION static std::size_t face_dof(std::size_t face,
                                                     std::size_t node) {
    return face * Basis::NNodes + node;
  }

  template <class ArrayU, class ArrayFlux>
  KOKKOS_INLINE_FUNCTION constexpr static void
  integral(std::size_t ielem, std::size_t jelem, ArrayU& du,
           const ArrayFlux& surface_flux, const BasisData& basis_data) {
    const T factor_1 = basis_data.boundary_interpolation_left[0];
    const T factor_2 =
        basis_data.boundary_interpolation_right[Basis::NNodes - 1];

    for (std::size_t jnode = 0; jnode < Basis::NNodes; ++jnode) {
      const std::size_t left_dof =
          DGSEM::utils::local_dof<Basis::NNodes>(0, jnode);
      const std::size_t right_dof =
          DGSEM::utils::local_dof<Basis::NNodes>(Basis::NNodes - 1, jnode);
      for (std::size_t var = 0; var < NVARS; ++var) {
        du(ielem, jelem, left_dof, var) -=
            surface_flux(ielem, jelem, face_dof(0, jnode), var) * factor_1;
        du(ielem, jelem, right_dof, var) +=
            surface_flux(ielem, jelem, face_dof(1, jnode), var) * factor_2;
      }
    }

    for (std::size_t inode = 0; inode < Basis::NNodes; ++inode) {
      const std::size_t bottom_dof =
          DGSEM::utils::local_dof<Basis::NNodes>(inode, 0);
      const std::size_t top_dof =
          DGSEM::utils::local_dof<Basis::NNodes>(inode, Basis::NNodes - 1);
      for (std::size_t var = 0; var < NVARS; ++var) {
        du(ielem, jelem, bottom_dof, var) -=
            surface_flux(ielem, jelem, face_dof(2, inode), var) * factor_1;
        du(ielem, jelem, top_dof, var) +=
            surface_flux(ielem, jelem, face_dof(3, inode), var) * factor_2;
      }
    }
  }
};

template <class Basis, class Equations, class T>
struct SurfaceIntegral<Basis, Equations, StructuredElementContainer<T, 3>> {
  using traits = equations::EquationTraits<Equations>;
  constexpr static std::size_t NVARS = traits::NVARS;
  using BasisData = typename Basis::DeviceData;

  KOKKOS_INLINE_FUNCTION static std::size_t face_dof(std::size_t face,
                                                     std::size_t node_i,
                                                     std::size_t node_j) {
    return face * Basis::NNodes * Basis::NNodes +
           DGSEM::utils::local_dof<Basis::NNodes>(node_i, node_j);
  }

  template <class ArrayU, class ArrayFlux>
  KOKKOS_INLINE_FUNCTION constexpr static void
  integral(std::size_t ielem, std::size_t jelem, std::size_t kelem, ArrayU& du,
           const ArrayFlux& surface_flux, const BasisData& basis_data) {
    const T factor_1 = basis_data.boundary_interpolation_left[0];
    const T factor_2 =
        basis_data.boundary_interpolation_right[Basis::NNodes - 1];

    for (std::size_t knode = 0; knode < Basis::NNodes; ++knode) {
      for (std::size_t jnode = 0; jnode < Basis::NNodes; ++jnode) {
        const std::size_t left_dof =
            DGSEM::utils::local_dof<Basis::NNodes>(0, jnode, knode);
        const std::size_t right_dof = DGSEM::utils::local_dof<Basis::NNodes>(
            Basis::NNodes - 1, jnode, knode);
        for (std::size_t var = 0; var < NVARS; ++var) {
          du(ielem, jelem, kelem, left_dof, var) -=
              surface_flux(ielem, jelem, kelem, face_dof(0, jnode, knode),
                           var) * factor_1;
          du(ielem, jelem, kelem, right_dof, var) +=
              surface_flux(ielem, jelem, kelem, face_dof(1, jnode, knode),
                           var) * factor_2;
        }
      }
    }

    for (std::size_t knode = 0; knode < Basis::NNodes; ++knode) {
      for (std::size_t inode = 0; inode < Basis::NNodes; ++inode) {
        const std::size_t bottom_dof =
            DGSEM::utils::local_dof<Basis::NNodes>(inode, 0, knode);
        const std::size_t top_dof = DGSEM::utils::local_dof<Basis::NNodes>(
            inode, Basis::NNodes - 1, knode);
        for (std::size_t var = 0; var < NVARS; ++var) {
          du(ielem, jelem, kelem, bottom_dof, var) -=
              surface_flux(ielem, jelem, kelem, face_dof(2, inode, knode),
                           var) * factor_1;
          du(ielem, jelem, kelem, top_dof, var) +=
              surface_flux(ielem, jelem, kelem, face_dof(3, inode, knode),
                           var) * factor_2;
        }
      }
    }

    for (std::size_t jnode = 0; jnode < Basis::NNodes; ++jnode) {
      for (std::size_t inode = 0; inode < Basis::NNodes; ++inode) {
        const std::size_t back_dof =
            DGSEM::utils::local_dof<Basis::NNodes>(inode, jnode, 0);
        const std::size_t front_dof = DGSEM::utils::local_dof<Basis::NNodes>(
            inode, jnode, Basis::NNodes - 1);
        for (std::size_t var = 0; var < NVARS; ++var) {
          du(ielem, jelem, kelem, back_dof, var) -=
              surface_flux(ielem, jelem, kelem, face_dof(4, inode, jnode),
                           var) * factor_1;
          du(ielem, jelem, kelem, front_dof, var) +=
              surface_flux(ielem, jelem, kelem, face_dof(5, inode, jnode),
                           var) * factor_2;
        }
      }
    }
  }
};
} // namespace DGSEM
