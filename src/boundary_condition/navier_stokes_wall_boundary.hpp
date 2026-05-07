#pragma once

#include <array>
#include <base/base.hpp>
#include <cstddef>
#include <equations/equations.hpp>
#include <utils/boundary_helper.hpp>

namespace DGSEM {
namespace detail {

template <class T>
KOKKOS_INLINE_FUNCTION std::array<T, 4>
reflect_normal_velocity_2d(const std::array<T, 4>& u_inner,
                           const std::array<T, 2>& normal) {
  const T rho = u_inner[0];
  const T vx = u_inner[1] / rho;
  const T vy = u_inner[2] / rho;
  const T normal_norm_sq = normal[0] * normal[0] + normal[1] * normal[1];
  const T vn = (vx * normal[0] + vy * normal[1]) / normal_norm_sq;
  const T vx_reflect = vx - static_cast<T>(2) * vn * normal[0];
  const T vy_reflect = vy - static_cast<T>(2) * vn * normal[1];

  return {rho, rho * vx_reflect, rho * vy_reflect, u_inner[3]};
}

template <class T>
KOKKOS_INLINE_FUNCTION void
remove_normal_gradient_velocity_2d(std::array<T, 4>& q,
                                   const std::array<T, 2>& normal) {
  const T normal_norm_sq = normal[0] * normal[0] + normal[1] * normal[1];
  const T vn = (q[1] * normal[0] + q[2] * normal[1]) / normal_norm_sq;
  q[1] -= vn * normal[0];
  q[2] -= vn * normal[1];
}

} // namespace detail

struct NSNoSlipAdiabaticWallBC {
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
    static_assert(NVARS == 4,
                  "NSNoSlipAdiabaticWallBC supports 2D compressible states.");

    const auto n_cells = mesh.get_num_cells();
    const std::size_t n_nodes = utils::infer_n_nodes<NDIMS>(u);

    utils::for_each_boundary_face_node<NDIMS>(
        face_id, index, n_cells, n_nodes, [&](const auto& location) {
          std::array<T, NVARS> u_inner{};
          for (std::size_t var = 0; var < NVARS; ++var) {
            u_inner[var] =
                u(location.ielem, location.jelem, location.boundary_dof, var);
          }

          const auto normal =
              utils::boundary_normal<NDIMS, T>(element_data, location);
          const auto u_wall =
              detail::reflect_normal_velocity_2d(u_inner, normal);
          const auto boundary_flux =
              location.left_state_inner
                  ? SurfaceFlux::numerical_flux(eq, u_inner, u_wall, normal)
                  : SurfaceFlux::numerical_flux(eq, u_wall, u_inner, normal);

          for (std::size_t var = 0; var < NVARS; ++var) {
            surface_flux(location.ielem, location.jelem, location.storage_dof,
                         var) = boundary_flux[var];
          }
        });
  }

  template <class Equations, class Mesh, class T, std::size_t NDIMS,
            class ArrayU, class ElementData, class ArrayFlux>
  KOKKOS_INLINE_FUNCTION void
  apply_gradient_device(const Mesh& mesh, const Equations& eq, const ArrayU& u,
                        const ElementData& element_data,
                        ArrayFlux& surface_flux, std::size_t face_id, T time,
                        int index = 0) const
    requires(NDIMS == 2)
  {
    using traits = equations::EquationTraits<Equations>;
    constexpr std::size_t NVARS = traits::NVARS;

    const auto n_cells = mesh.get_num_cells();
    const std::size_t n_nodes = utils::infer_n_nodes<NDIMS>(u);

    utils::for_each_boundary_face_node<NDIMS>(
        face_id, index, n_cells, n_nodes, [&](const auto& location) {
          std::array<T, NVARS> u_inner{};
          for (std::size_t var = 0; var < NVARS; ++var) {
            u_inner[var] =
                u(location.ielem, location.jelem, location.boundary_dof, var);
          }

          auto q_wall = eq.gradient_variables(u_inner);
          q_wall[1] = T{0};
          q_wall[2] = T{0};

          for (std::size_t var = 0; var < Equations::NGRAD_VARS; ++var) {
            surface_flux(location.ielem, location.jelem, location.storage_dof,
                         var) = q_wall[var];
          }
        });
  }

  template <class Equations, class Mesh, class T, std::size_t NDIMS,
            class ArrayU, class ElementData, class ArrayViscousFlux,
            class ArrayFlux>
  KOKKOS_INLINE_FUNCTION void apply_viscous_device(
      const Mesh& mesh, const Equations& eq, const ArrayU& u,
      const ElementData& element_data, const ArrayViscousFlux& viscous_flux,
      ArrayFlux& surface_flux, std::size_t face_id, T time, int index = 0) const
    requires(NDIMS == 2)
  {
    const auto n_cells = mesh.get_num_cells();
    const std::size_t n_nodes = utils::infer_n_nodes<NDIMS>(u);

    utils::for_each_boundary_face_node<NDIMS>(
        face_id, index, n_cells, n_nodes, [&](const auto& location) {
          const auto normal =
              utils::boundary_normal<NDIMS, T>(element_data, location);

          surface_flux(location.ielem, location.jelem, location.storage_dof,
                       0) = T{0};
          for (std::size_t var = 1; var < Equations::NVARS - 1; ++var) {
            surface_flux(location.ielem, location.jelem, location.storage_dof,
                         var) = viscous_flux(location.ielem, location.jelem,
                                             location.boundary_dof, var, 0) *
                                    normal[0] +
                                viscous_flux(location.ielem, location.jelem,
                                             location.boundary_dof, var, 1) *
                                    normal[1];
          }
          surface_flux(location.ielem, location.jelem, location.storage_dof,
                       Equations::NVARS - 1) = T{0};
        });
  }
};

struct NSSlipAdiabaticWallBC {
  template <class Equations, class SurfaceFlux, class Mesh, class T,
            std::size_t NDIMS, class ArrayU, class ElementData, class ArrayFlux>
  KOKKOS_INLINE_FUNCTION void
  apply_device(const Mesh& mesh, const Equations& eq, const ArrayU& u,
               const ElementData& element_data, ArrayFlux& surface_flux,
               std::size_t face_id, T time, int index = 0) const
    requires(NDIMS == 2)
  {
    NSNoSlipAdiabaticWallBC{}
        .template apply_device<Equations, SurfaceFlux, Mesh, T, NDIMS>(
            mesh, eq, u, element_data, surface_flux, face_id, time, index);
  }

  template <class Equations, class Mesh, class T, std::size_t NDIMS,
            class ArrayU, class ElementData, class ArrayFlux>
  KOKKOS_INLINE_FUNCTION void
  apply_gradient_device(const Mesh& mesh, const Equations& eq, const ArrayU& u,
                        const ElementData& element_data,
                        ArrayFlux& surface_flux, std::size_t face_id, T time,
                        int index = 0) const
    requires(NDIMS == 2)
  {
    using traits = equations::EquationTraits<Equations>;
    constexpr std::size_t NVARS = traits::NVARS;

    const auto n_cells = mesh.get_num_cells();
    const std::size_t n_nodes = utils::infer_n_nodes<NDIMS>(u);

    utils::for_each_boundary_face_node<NDIMS>(
        face_id, index, n_cells, n_nodes, [&](const auto& location) {
          std::array<T, NVARS> u_inner{};
          for (std::size_t var = 0; var < NVARS; ++var) {
            u_inner[var] =
                u(location.ielem, location.jelem, location.boundary_dof, var);
          }

          auto q_wall = eq.gradient_variables(u_inner);
          const auto normal =
              utils::boundary_normal<NDIMS, T>(element_data, location);
          detail::remove_normal_gradient_velocity_2d(q_wall, normal);

          for (std::size_t var = 0; var < Equations::NGRAD_VARS; ++var) {
            surface_flux(location.ielem, location.jelem, location.storage_dof,
                         var) = q_wall[var];
          }
        });
  }

  template <class Equations, class Mesh, class T, std::size_t NDIMS,
            class ArrayU, class ElementData, class ArrayViscousFlux,
            class ArrayFlux>
  KOKKOS_INLINE_FUNCTION void apply_viscous_device(
      const Mesh& mesh, const Equations& eq, const ArrayU& u,
      const ElementData& element_data, const ArrayViscousFlux& viscous_flux,
      ArrayFlux& surface_flux, std::size_t face_id, T time, int index = 0) const
    requires(NDIMS == 2)
  {
    const auto n_cells = mesh.get_num_cells();
    const std::size_t n_nodes = utils::infer_n_nodes<NDIMS>(u);

    utils::for_each_boundary_face_node<NDIMS>(
        face_id, index, n_cells, n_nodes, [&](const auto& location) {
          for (std::size_t var = 0; var < Equations::NVARS; ++var) {
            surface_flux(location.ielem, location.jelem, location.storage_dof,
                         var) = T{0};
          }
        });
  }
};

} // namespace DGSEM
