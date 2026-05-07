#pragma once

#include <array>
#include <base/base.hpp>
#include <cstddef>
#include <equations/equations.hpp>
#include <type_traits>
#include <utils/boundary_helper.hpp>

namespace DGSEM {
template <class Func>
struct DirichletBC {
  Func func;

  explicit DirichletBC(Func func_) : func(func_) {}

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

    auto get_u_outer_device = [&](const std::array<T, NDIMS>& coord, T t) {
      std::array<T, NVARS> u_out{};
      if constexpr (std::is_invocable_v<Func, const std::array<T, NDIMS>&, T>) {
        u_out = func(coord, t);
      } else {
        for (std::size_t var = 0; var < NVARS; ++var) {
          u_out[var] = func[var];
        }
      }
      return u_out;
    };

    const std::size_t n_cells = mesh.get_num_cells(0);
    const std::size_t n_nodes = u.extent(1);

    if (face_id == 0) {
      std::array<T, NVARS> u_boundary{};
      for (std::size_t var = 0; var < NVARS; ++var) {
        u_boundary[var] = u(0, 0, var);
      }
      const std::array<T, 1> coord{mesh.get_face_coord(0)};
      const auto u_outer = get_u_outer_device(coord, time);
      const auto boundary_flux =
          SurfaceFlux::numerical_flux(eq, u_outer, u_boundary);
      for (std::size_t var = 0; var < NVARS; ++var) {
        surface_flux(0, 0, var) = boundary_flux[var];
      }
    } else {
      std::array<T, NVARS> u_boundary{};
      for (std::size_t var = 0; var < NVARS; ++var) {
        u_boundary[var] = u(n_cells - 1, n_nodes - 1, var);
      }
      const std::array<T, 1> coord{mesh.get_face_coord(1)};
      const auto u_outer = get_u_outer_device(coord, time);
      const auto boundary_flux =
          SurfaceFlux::numerical_flux(eq, u_boundary, u_outer);
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

    auto get_u_outer_device = [&](const std::array<T, NDIMS>& coord, T t) {
      std::array<T, NVARS> u_out{};
      if constexpr (std::is_invocable_v<Func, const std::array<T, NDIMS>&, T>) {
        u_out = func(coord, t);
      } else {
        for (std::size_t var = 0; var < NVARS; ++var) {
          u_out[var] = func[var];
        }
      }
      return u_out;
    };

    const auto n_cells = mesh.get_num_cells();
    const std::size_t n_nodes = utils::infer_n_nodes<NDIMS>(u);

    utils::for_each_boundary_face_node<NDIMS>(
        face_id, index, n_cells, n_nodes,
        [&](const auto& location) {
          const auto coord =
              utils::boundary_coord<NDIMS, T>(element_data, location);
          const auto u_outer = get_u_outer_device(coord, time);
          const auto normal = utils::boundary_normal<NDIMS, T>(element_data,
                                                               location);

          std::array<T, NVARS> u_boundary{};
          for (std::size_t var = 0; var < NVARS; ++var) {
            u_boundary[var] =
                u(location.ielem, location.jelem, location.boundary_dof, var);
          }

          const auto boundary_flux =
              location.left_state_inner
                  ? SurfaceFlux::numerical_flux(eq, u_boundary, u_outer, normal)
                  : SurfaceFlux::numerical_flux(eq, u_outer, u_boundary, normal);
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

    auto get_u_outer_device = [&](const std::array<T, NDIMS>& coord, T t) {
      std::array<T, NVARS> u_out{};
      if constexpr (std::is_invocable_v<Func, const std::array<T, NDIMS>&, T>) {
        u_out = func(coord, t);
      } else {
        for (std::size_t var = 0; var < NVARS; ++var) {
          u_out[var] = func[var];
        }
      }
      return u_out;
    };

    const auto n_cells = mesh.get_num_cells();
    const std::size_t n_nodes = utils::infer_n_nodes<NDIMS>(u);

    utils::for_each_boundary_face_node<NDIMS>(
        face_id, index, n_cells, n_nodes,
        [&](const auto& location) {
          const auto coord =
              utils::boundary_coord<NDIMS, T>(element_data, location);
          const auto u_outer = get_u_outer_device(coord, time);
          const auto q_outer = eq.gradient_variables(u_outer);

          for (std::size_t var = 0; var < Equations::NGRAD_VARS; ++var) {
            surface_flux(location.ielem, location.jelem, location.storage_dof,
                         var) = q_outer[var];
          }
        });
  }

  template <class Equations, class Mesh, class T, std::size_t NDIMS,
            class ArrayU, class ElementData, class ArrayViscousFlux,
            class ArrayFlux>
  KOKKOS_INLINE_FUNCTION void
  apply_viscous_device(const Mesh& mesh, const Equations& eq, const ArrayU& u,
                       const ElementData& element_data,
                       const ArrayViscousFlux& viscous_flux,
                       ArrayFlux& surface_flux, std::size_t face_id, T time,
                       int index = 0) const
    requires(NDIMS == 2)
  {
    const auto n_cells = mesh.get_num_cells();
    const std::size_t n_nodes = utils::infer_n_nodes<NDIMS>(u);

    utils::for_each_boundary_face_node<NDIMS>(
        face_id, index, n_cells, n_nodes,
        [&](const auto& location) {
          const auto normal =
              utils::boundary_normal<NDIMS, T>(element_data, location);

          for (std::size_t var = 0; var < Equations::NVARS; ++var) {
            surface_flux(location.ielem, location.jelem, location.storage_dof,
                         var) =
                viscous_flux(location.ielem, location.jelem,
                             location.boundary_dof, var, 0) *
                    normal[0] +
                viscous_flux(location.ielem, location.jelem,
                             location.boundary_dof, var, 1) *
                    normal[1];
          }
        });
  }
};

template <class Func, class Predicate>
struct MixedDirichletSlipWallBC {
  Func func;
  Predicate use_dirichlet;

  MixedDirichletSlipWallBC(Func func_, Predicate use_dirichlet_)
      : func(func_), use_dirichlet(use_dirichlet_) {}

  template <class T>
  KOKKOS_INLINE_FUNCTION static std::array<T, 4>
  reflect_state(const std::array<T, 4>& u_inner,
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

    static_assert(NVARS == 4, "MixedDirichletSlipWallBC currently supports 2D "
                              "compressible Euler.");

    const auto n_cells = mesh.get_num_cells();
    const std::size_t n_nodes = utils::infer_n_nodes<NDIMS>(u);

    utils::for_each_boundary_face_node<NDIMS>(
        face_id, index, n_cells, n_nodes,
        [&](const auto& location) {
          std::array<T, NVARS> u_inner{};
          for (std::size_t var = 0; var < NVARS; ++var) {
            u_inner[var] =
                u(location.ielem, location.jelem, location.boundary_dof, var);
          }

          const auto coord =
              utils::boundary_coord<NDIMS, T>(element_data, location);
          const auto normal =
              utils::boundary_normal<NDIMS, T>(element_data, location);

          const auto boundary_flux = [&]() {
            if (use_dirichlet(coord, time)) {
              const auto u_outer = func(coord, time);
              return location.left_state_inner
                         ? SurfaceFlux::numerical_flux(eq, u_inner, u_outer,
                                                       normal)
                         : SurfaceFlux::numerical_flux(eq, u_outer, u_inner,
                                                       normal);
            }

            const auto u_wall = reflect_state(u_inner, normal);
            return location.left_state_inner
                       ? SurfaceFlux::numerical_flux(eq, u_inner, u_wall, normal)
                       : SurfaceFlux::numerical_flux(eq, u_wall, u_inner,
                                                     normal);
          }();

          for (std::size_t var = 0; var < NVARS; ++var) {
            surface_flux(location.ielem, location.jelem, location.storage_dof,
                         var) = boundary_flux[var];
          }
        });
  }
};
} // namespace DGSEM
