#pragma once

#include <array>
#include <containers/containers.hpp>
#include <cstddef>
#include <equations/equations.hpp>
#include <utils/local_dof.hpp>

namespace DGSEM {

template <class Basis, class Equations, class SurfaceFlux, class Element>
struct InterfaceHelper;

template <class Basis, class Equations, class SurfaceFlux, class T>
struct InterfaceHelper<Basis, Equations, SurfaceFlux,
                       StructuredElementContainer<T, 1>> {
  using traits = equations::EquationTraits<Equations>;

  constexpr static std::size_t NVARS = traits::NVARS;

  template <class IndexArray, class ArrayU, class ArrayFlux>
  KOKKOS_INLINE_FUNCTION static void
  interface_flux(const IndexArray& left_neighbors, const Equations& eq,
                 std::size_t ielem, const ArrayU& u, ArrayFlux& surface_flux) {
    std::size_t left_elem = left_neighbors(ielem, 0);
    if (left_elem != static_cast<std::size_t>(-1)) {
      std::array<T, NVARS> u_ll{};
      std::array<T, NVARS> u_rr{};
      for (std::size_t var = 0; var < NVARS; ++var) {
        u_ll[var] = u(left_elem, Basis::NNodes - 1, var);
        u_rr[var] = u(ielem, 0, var);
      }

      std::array<T, NVARS> interface =
          SurfaceFlux::numerical_flux(eq, u_ll, u_rr);

      for (std::size_t var = 0; var < NVARS; ++var) {
        surface_flux(left_elem, 1, var) = interface[var];
        surface_flux(ielem, 0, var) = interface[var];
      }
    }
  }
};

template <class Basis, class Equations, class SurfaceFlux, class T>
struct InterfaceHelper<Basis, Equations, SurfaceFlux,
                       StructuredElementContainer<T, 2>> {
  using traits = equations::EquationTraits<Equations>;
  constexpr static std::size_t NVARS = traits::NVARS;
  using MetricArray = typename jacobian_type_traits<T, 2>::JacobianMatrix;
  using ScalarArray = typename scalar_node_type_traits<T, 2>::ScalarArray;

  KOKKOS_INLINE_FUNCTION static std::size_t face_dof(std::size_t face,
                                                     std::size_t node) {
    return face * Basis::NNodes + node;
  }

  template <class IndexArray, class ArrayU, class ArrayFlux>
  KOKKOS_INLINE_FUNCTION static void
  interface_flux(const IndexArray& neighbors,
                 const MetricArray& contravariant_vectors,
                 const ScalarArray& inverse_jacobian, const Equations& eq,
                 std::size_t ielem, std::size_t jelem, const ArrayU& u,
                 ArrayFlux& surface_flux) {
    const std::size_t left_i = neighbors(ielem, jelem, 0, 0);
    const std::size_t left_j = neighbors(ielem, jelem, 0, 1);
    if (left_i != static_cast<std::size_t>(-1) &&
        left_j != static_cast<std::size_t>(-1)) {
      for (std::size_t jnode = 0; jnode < Basis::NNodes; ++jnode) {
        std::array<T, NVARS> u_ll{};
        std::array<T, NVARS> u_rr{};
        const std::size_t left_dof =
            DGSEM::utils::local_dof<Basis::NNodes>(Basis::NNodes - 1, jnode);
        const std::size_t right_dof =
            DGSEM::utils::local_dof<Basis::NNodes>(0, jnode);
        const T sign_jacobian =
            inverse_jacobian(ielem, jelem, right_dof) >= T{0} ? T{1} : T{-1};
        const std::array<T, 2> normal = {
            sign_jacobian *
                contravariant_vectors(ielem, jelem, right_dof, 0, 0),
            sign_jacobian *
                contravariant_vectors(ielem, jelem, right_dof, 0, 1)};
        for (std::size_t var = 0; var < NVARS; ++var) {
          u_ll[var] = u(left_i, left_j, left_dof, var);
          u_rr[var] = u(ielem, jelem, right_dof, var);
        }

        auto interface = SurfaceFlux::numerical_flux(eq, u_ll, u_rr, normal);
        for (std::size_t var = 0; var < NVARS; ++var) {
          surface_flux(left_i, left_j, face_dof(1, jnode), var) =
              interface[var];
          surface_flux(ielem, jelem, face_dof(0, jnode), var) = interface[var];
        }
      }
    }

    const std::size_t bottom_i = neighbors(ielem, jelem, 1, 0);
    const std::size_t bottom_j = neighbors(ielem, jelem, 1, 1);
    if (bottom_i != static_cast<std::size_t>(-1) &&
        bottom_j != static_cast<std::size_t>(-1)) {
      for (std::size_t inode = 0; inode < Basis::NNodes; ++inode) {
        std::array<T, NVARS> u_ll{};
        std::array<T, NVARS> u_rr{};
        const std::size_t bottom_dof =
            DGSEM::utils::local_dof<Basis::NNodes>(inode, Basis::NNodes - 1);
        const std::size_t top_dof =
            DGSEM::utils::local_dof<Basis::NNodes>(inode, 0);
        const T sign_jacobian =
            inverse_jacobian(ielem, jelem, top_dof) >= T{0} ? T{1} : T{-1};
        const std::array<T, 2> normal = {
            sign_jacobian * contravariant_vectors(ielem, jelem, top_dof, 1, 0),
            sign_jacobian * contravariant_vectors(ielem, jelem, top_dof, 1, 1)};
        for (std::size_t var = 0; var < NVARS; ++var) {
          u_ll[var] = u(bottom_i, bottom_j, bottom_dof, var);
          u_rr[var] = u(ielem, jelem, top_dof, var);
        }

        auto interface = SurfaceFlux::numerical_flux(eq, u_ll, u_rr, normal);
        for (std::size_t var = 0; var < NVARS; ++var) {
          surface_flux(bottom_i, bottom_j, face_dof(3, inode), var) =
              interface[var];
          surface_flux(ielem, jelem, face_dof(2, inode), var) = interface[var];
        }
      }
    }
  }
};

} // namespace DGSEM
