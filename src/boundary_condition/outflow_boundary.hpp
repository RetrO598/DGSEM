#pragma once

#include <array>
#include <base/base.hpp>
#include <cstddef>
#include <equations/equations.hpp>
#include <utils/boundary_helper.hpp>

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

    const std::size_t n_cells = mesh.get_num_cells(0);
    const std::size_t n_nodes = u.extent(1);

    if (face_id == 0) {
      std::array<T, NVARS> u_left{};
      std::array<T, NVARS> u_right{};
      for (std::size_t var = 0; var < NVARS; ++var) {
        u_left[var] = u(0, 0, var);
        for (std::size_t node = 0; node < n_nodes; ++node) {
          u_right[var] += u(0, node, var);
        }
        u_right[var] /= static_cast<T>(n_nodes);
      }
      const auto boundary_flux = SurfaceFlux::numerical_flux(eq, u_left, u_right);
      for (std::size_t var = 0; var < NVARS; ++var) {
        surface_flux(0, 0, var) = boundary_flux[var];
      }
    } else {
      std::array<T, NVARS> u_left{};
      std::array<T, NVARS> u_right{};
      for (std::size_t var = 0; var < NVARS; ++var) {
        u_left[var] = u(n_cells - 1, n_nodes - 1, var);
        for (std::size_t node = 0; node < n_nodes; ++node) {
          u_right[var] += u(n_cells - 1, node, var);
        }
        u_right[var] /= static_cast<T>(n_nodes);
      }
      const auto boundary_flux = SurfaceFlux::numerical_flux(eq, u_left, u_right);
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
    const std::size_t n_nodes = utils::infer_n_nodes<NDIMS>(u);

    utils::for_each_boundary_face_node<NDIMS>(
        face_id, index, n_cells, n_nodes,
        [&](const auto& location) {
          std::array<T, NVARS> u_left{};
          std::array<T, NVARS> u_right{};
          for (std::size_t var = 0; var < NVARS; ++var) {
            u_left[var] =
                u(location.ielem, location.jelem, location.boundary_dof, var);
            if (location.dim == 0) {
              const std::size_t jnode = location.boundary_dof / n_nodes;
              for (std::size_t inode = 0; inode < n_nodes; ++inode) {
                u_right[var] +=
                    u(location.ielem, location.jelem,
                      utils::local_dof(n_nodes, inode, jnode), var);
              }
            } else {
              const std::size_t inode = location.boundary_dof % n_nodes;
              for (std::size_t jnode = 0; jnode < n_nodes; ++jnode) {
                u_right[var] +=
                    u(location.ielem, location.jelem,
                      utils::local_dof(n_nodes, inode, jnode), var);
              }
            }
            u_right[var] /= static_cast<T>(n_nodes);
          }

          const auto normal =
              utils::boundary_normal<NDIMS, T>(element_data, location);
          const auto boundary_flux =
              location.left_state_inner
                  ? SurfaceFlux::numerical_flux(eq, u_left, u_right, normal)
                  : SurfaceFlux::numerical_flux(eq, u_right, u_left, normal);
          for (std::size_t var = 0; var < NVARS; ++var) {
            surface_flux(location.ielem, location.jelem, location.storage_dof,
                         var) = boundary_flux[var];
          }
        });
  }
};
} // namespace DGSEM
