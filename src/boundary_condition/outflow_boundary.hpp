#pragma once

#include <array>
#include <base/base.hpp>
#include <cstddef>
#include <equations/equations.hpp>
#include <utils/local_dof.hpp>

namespace DGSEM {
struct OutflowBC {

  template <class Equations, class SurfaceFlux, class Mesh, class T,
            std::size_t NDIMS, class ArrayU, class ElementData, class ArrayFlux>
  KOKKOS_INLINE_FUNCTION void
  apply_device(const Mesh& mesh, const Equations& eq, const ArrayU& u,
               const ElementData& element_data, ArrayFlux& surface_flux,
               std::size_t face_id, T time, int index = 0) const
    requires(NDIMS == 1)
  {
    using traits = equations::EquationTraits<Equations>;
    constexpr std::size_t NVARS = traits::NVARS;

    std::size_t n_cells = mesh.get_num_cells(0);
    std::size_t n_nodes = u.extent(1);

    if (face_id == 0) {
      std::array<T, NVARS> u_left{};
      std::array<T, NVARS> u_right{};
      for (std::size_t var = 0; var < NVARS; ++var) {
        u_left[var] = u(0, 0, var);
        u_right[var] = 0;
        for (std::size_t node = 0; node < n_nodes; ++node) {
          u_right[var] += u(0, node, var);
        }
        u_right[var] /= static_cast<T>(n_nodes);
      }
      auto boundary_flux = SurfaceFlux::numerical_flux(eq, u_left, u_right);
      for (std::size_t var = 0; var < NVARS; ++var) {
        surface_flux(0, 0, var) = boundary_flux[var];
      }
    } else {
      std::array<T, NVARS> u_left{};
      std::array<T, NVARS> u_right{};
      for (std::size_t var = 0; var < NVARS; ++var) {
        u_left[var] = u(n_cells - 1, n_nodes - 1, var);
        u_right[var] = 0;
        for (std::size_t node = 0; node < n_nodes; ++node) {
          u_right[var] += u(n_cells - 1, node, var);
        }
        u_right[var] /= static_cast<T>(n_nodes);
      }
      auto boundary_flux = SurfaceFlux::numerical_flux(eq, u_left, u_right);
      for (std::size_t var = 0; var < NVARS; ++var) {
        surface_flux(n_cells - 1, 1, var) = boundary_flux[var];
      }
    }
  }

  template <class Equations, class SurfaceFlux, class Mesh, class T,
            std::size_t NDIMS, class ArrayU, class ElementData, class ArrayFlux>
  KOKKOS_INLINE_FUNCTION void
  apply_device(const Mesh& mesh, const Equations& eq, const ArrayU& u,
               const ElementData& element_data, ArrayFlux& surface_flux,
               std::size_t face_id, T time, int index = 0) const
    requires(NDIMS == 2)
  {
    using traits = equations::EquationTraits<Equations>;
    constexpr std::size_t NVARS = traits::NVARS;

    const auto n_cells = mesh.get_num_cells();
    std::size_t n_nodes = 0;
    while (n_nodes * n_nodes < u.extent(2)) {
      ++n_nodes;
    }
    auto face_dof = [&](std::size_t face, std::size_t node) {
      return face * n_nodes + node;
    };
    auto local_dof = [&](std::size_t inode, std::size_t jnode) {
      return jnode * n_nodes + inode;
    };
    auto get_normal = [&](std::size_t ielem, std::size_t jelem, std::size_t dof,
                          std::size_t dim) {
      const T sign_jacobian =
          element_data.inverse_jacobian(ielem, jelem, dof) >= T{0} ? T{1}
                                                                   : T{-1};
      return std::array<T, 2>{
          sign_jacobian *
              element_data.contravariant_vectors(ielem, jelem, dof, dim, 0),
          sign_jacobian *
              element_data.contravariant_vectors(ielem, jelem, dof, dim, 1)};
    };

    if (face_id == 0) {
      const std::size_t ielem = 0;
      const std::size_t jelem = static_cast<std::size_t>(index);
      for (std::size_t jnode = 0; jnode < n_nodes; ++jnode) {
        std::array<T, NVARS> u_left{};
        std::array<T, NVARS> u_right{};
        for (std::size_t var = 0; var < NVARS; ++var) {
          u_left[var] = u(ielem, jelem, local_dof(0, jnode), var);
          u_right[var] = 0;
          for (std::size_t inode = 0; inode < n_nodes; ++inode) {
            u_right[var] += u(ielem, jelem, local_dof(inode, jnode), var);
          }
          u_right[var] /= static_cast<T>(n_nodes);
        }
        const std::size_t boundary_dof = local_dof(0, jnode);
        const auto normal = get_normal(ielem, jelem, boundary_dof, 0);
        const auto boundary_flux =
            SurfaceFlux::numerical_flux(eq, u_right, u_left, normal);
        for (std::size_t var = 0; var < NVARS; ++var) {
          surface_flux(ielem, jelem, face_dof(0, jnode), var) =
              boundary_flux[var];
        }
      }
    } else if (face_id == 1) {
      const std::size_t ielem = n_cells[0] - 1;
      const std::size_t jelem = static_cast<std::size_t>(index);
      for (std::size_t jnode = 0; jnode < n_nodes; ++jnode) {
        std::array<T, NVARS> u_left{};
        std::array<T, NVARS> u_right{};
        for (std::size_t var = 0; var < NVARS; ++var) {
          u_left[var] = u(ielem, jelem, local_dof(n_nodes - 1, jnode), var);
          u_right[var] = 0;
          for (std::size_t inode = 0; inode < n_nodes; ++inode) {
            u_right[var] += u(ielem, jelem, local_dof(inode, jnode), var);
          }
          u_right[var] /= static_cast<T>(n_nodes);
        }
        const std::size_t boundary_dof = local_dof(n_nodes - 1, jnode);
        const auto normal = get_normal(ielem, jelem, boundary_dof, 0);
        const auto boundary_flux =
            SurfaceFlux::numerical_flux(eq, u_left, u_right, normal);
        for (std::size_t var = 0; var < NVARS; ++var) {
          surface_flux(ielem, jelem, face_dof(1, jnode), var) =
              boundary_flux[var];
        }
      }
    } else if (face_id == 2) {
      const std::size_t ielem = static_cast<std::size_t>(index);
      const std::size_t jelem = 0;
      for (std::size_t inode = 0; inode < n_nodes; ++inode) {
        std::array<T, NVARS> u_left{};
        std::array<T, NVARS> u_right{};
        for (std::size_t var = 0; var < NVARS; ++var) {
          u_left[var] = u(ielem, jelem, local_dof(inode, 0), var);
          u_right[var] = 0;
          for (std::size_t jnode = 0; jnode < n_nodes; ++jnode) {
            u_right[var] += u(ielem, jelem, local_dof(inode, jnode), var);
          }
          u_right[var] /= static_cast<T>(n_nodes);
        }
        const std::size_t boundary_dof = local_dof(inode, 0);
        const auto normal = get_normal(ielem, jelem, boundary_dof, 1);
        const auto boundary_flux =
            SurfaceFlux::numerical_flux(eq, u_right, u_left, normal);
        for (std::size_t var = 0; var < NVARS; ++var) {
          surface_flux(ielem, jelem, face_dof(2, inode), var) =
              boundary_flux[var];
        }
      }
    } else {
      const std::size_t ielem = static_cast<std::size_t>(index);
      const std::size_t jelem = n_cells[1] - 1;
      for (std::size_t inode = 0; inode < n_nodes; ++inode) {
        std::array<T, NVARS> u_left{};
        std::array<T, NVARS> u_right{};
        for (std::size_t var = 0; var < NVARS; ++var) {
          u_left[var] = u(ielem, jelem, local_dof(inode, n_nodes - 1), var);
          u_right[var] = 0;
          for (std::size_t jnode = 0; jnode < n_nodes; ++jnode) {
            u_right[var] += u(ielem, jelem, local_dof(inode, jnode), var);
          }
          u_right[var] /= static_cast<T>(n_nodes);
        }
        const std::size_t boundary_dof = local_dof(inode, n_nodes - 1);
        const auto normal = get_normal(ielem, jelem, boundary_dof, 1);
        const auto boundary_flux =
            SurfaceFlux::numerical_flux(eq, u_left, u_right, normal);
        for (std::size_t var = 0; var < NVARS; ++var) {
          surface_flux(ielem, jelem, face_dof(3, inode), var) =
              boundary_flux[var];
        }
      }
    }
  }
};
} // namespace DGSEM