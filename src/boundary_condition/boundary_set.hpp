#pragma once

#include <array>
#include <base/base.hpp>
#include <cstddef>
#include <equations/equations.hpp>
#include <tuple>
#include <type_traits>
#include <utils/local_dof.hpp>

namespace DGSEM {

template <std::size_t NDIMS, class BC, class Equations, class SurfaceFlux,
          class Mesh, class T, class ArrayU, class ElementData, class ArrayFlux>
struct BCDispatcher;

template <class BC, class Equations, class SurfaceFlux, class Mesh, class T,
          class ArrayU, class ElementData, class ArrayFlux>
struct BCDispatcher<1, BC, Equations, SurfaceFlux, Mesh, T, ArrayU,
                    ElementData, ArrayFlux> {
  static void dispatch(const BC& bc, const Mesh& mesh, const Equations& eq,
                       const ArrayU& u, const ElementData& element_data,
                       ArrayFlux& surface_flux, std::size_t face_id, T time) {
    // 1D case: No parallel_for needed for many elements,
    // but we use a single-threaded kernel to access device data.
    Kokkos::parallel_for(
        "BC_Apply_1D", 1, KOKKOS_LAMBDA(int) {
          bc.template apply_device<Equations, SurfaceFlux, Mesh, T, 1>(
              mesh, eq, u, element_data, surface_flux, face_id, time);
        });
  }
};

template <class BC, class Equations, class SurfaceFlux, class Mesh, class T,
          class ArrayU, class ElementData, class ArrayFlux>
struct BCDispatcher<2, BC, Equations, SurfaceFlux, Mesh, T, ArrayU,
                    ElementData, ArrayFlux> {
  static void dispatch(const BC& bc, const Mesh& mesh, const Equations& eq,
                       const ArrayU& u, const ElementData& element_data,
                       ArrayFlux& surface_flux, std::size_t face_id, T time) {
    auto n_cells = mesh.get_num_cells();
    std::size_t range = (face_id < 2) ? n_cells[1] : n_cells[0];

    Kokkos::parallel_for(
        "BC_Apply_2D", range, KOKKOS_LAMBDA(int i) {
          bc.template apply_device<Equations, SurfaceFlux, Mesh, T, 2>(
              mesh, eq, u, element_data, surface_flux, face_id, time, i);
        });
  }
};

template <class... FaceBCs>
struct BoundarySet {
  static constexpr std::size_t NFACES = sizeof...(FaceBCs);

  std::tuple<FaceBCs...> faces_;

  BoundarySet(FaceBCs... bcs) : faces_(std::move(bcs)...) {}

  template <std::size_t I = 0, typename F>
  void tuple_switch(std::size_t idx, F&& f) const {
    if constexpr (I < NFACES) {
      if (I == idx) {
        f(std::get<I>(faces_));
      } else {
        tuple_switch<I + 1>(idx, std::forward<F>(f));
      }
    } else {
      // throw std::out_of_range("BoundarySet::apply: face_id out of range");
    }
  }

  // Kokkos apply
  template <class Equations, class SurfaceFlux, class Mesh, class T,
            std::size_t NDIMS, class ArrayU, class ElementData, class ArrayFlux>
  void apply(const Mesh& mesh, const Equations& eq, const ArrayU& u,
             const ElementData& element_data, ArrayFlux& surface_flux,
             std::size_t face_id, T time) const {
    if constexpr (NFACES > 0) {
      tuple_switch(face_id, [&](const auto& bc) {
        BCDispatcher<NDIMS, std::decay_t<decltype(bc)>, Equations, SurfaceFlux,
                     Mesh, T, ArrayU, ElementData, ArrayFlux>::dispatch(
            bc, mesh, eq, u, element_data, surface_flux, face_id, time);
      });
    }
  }
};

struct PeriodicBC {

  template <class Equations, class SurfaceFlux, class Mesh, class T,
            std::size_t NDIMS, class ArrayU, class ElementData, class ArrayFlux>
  KOKKOS_INLINE_FUNCTION void
  apply_device(const Mesh& mesh, const Equations& eq, const ArrayU& u,
               const ElementData& element_data, ArrayFlux& surface_flux,
               std::size_t face_id, T time, int index = 0) const {
    // Periodic BCs are usually handled by the interface flux logic with
    // wrapping
    return;
  }
};

struct OutflowBC {
  template <class Equations, class SurfaceFlux, class Mesh, class T,
            std::size_t NDIMS, class ArrayU, class ElementData, class ArrayFlux>
  KOKKOS_INLINE_FUNCTION void
  apply_device(const Mesh& mesh, const Equations& eq, const ArrayU& u,
               const ElementData& element_data, ArrayFlux& surface_flux,
               std::size_t face_id, T time, int index = 0) const {
    using traits = equations::EquationTraits<Equations>;
    constexpr std::size_t NVARS = traits::NVARS;

    if constexpr (NDIMS == 1) {
      std::size_t n_cells = mesh.get_num_cells(0);
      std::size_t n_nodes = u.extent(1);

      if (face_id == 0) {
        std::array<T, NVARS> u_boundary{};
        for (std::size_t var = 0; var < NVARS; ++var) {
          u_boundary[var] = u(0, 0, var);
        }
        auto boundary_flux =
            SurfaceFlux::numerical_flux(eq, u_boundary, u_boundary);
        for (std::size_t var = 0; var < NVARS; ++var) {
          surface_flux(0, 0, var) = boundary_flux[var];
        }
      } else {
        std::array<T, NVARS> u_boundary{};
        for (std::size_t var = 0; var < NVARS; ++var) {
          u_boundary[var] = u(n_cells - 1, n_nodes - 1, var);
        }
        auto boundary_flux =
            SurfaceFlux::numerical_flux(eq, u_boundary, u_boundary);
        for (std::size_t var = 0; var < NVARS; ++var) {
          surface_flux(n_cells - 1, 1, var) = boundary_flux[var];
        }
      }
    } else if constexpr (NDIMS == 2) {
      constexpr std::size_t NVARS_LOCAL = traits::NVARS;
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
      auto get_normal = [&](std::size_t ielem, std::size_t jelem,
                            std::size_t dof, std::size_t dim) {
        const T sign_jacobian =
            element_data.inverse_jacobian(ielem, jelem, dof) >= T{0} ? T{1}
                                                                      : T{-1};
        return std::array<T, 2>{
            sign_jacobian *
                element_data.contravariant_vectors(ielem, jelem, dof, dim, 0),
            sign_jacobian *
                element_data.contravariant_vectors(ielem, jelem, dof, dim, 1)};
      };

      auto write_flux = [&](std::size_t ielem, std::size_t jelem,
                            std::size_t boundary_dof, std::size_t storage_dof,
                            std::size_t dim, bool left_state_inner) {
        std::array<T, NVARS_LOCAL> u_inner{};
        for (std::size_t var = 0; var < NVARS_LOCAL; ++var) {
          u_inner[var] = u(ielem, jelem, boundary_dof, var);
        }
        const auto normal = get_normal(ielem, jelem, boundary_dof, dim);
        const auto boundary_flux =
            left_state_inner
                ? SurfaceFlux::numerical_flux(eq, u_inner, u_inner, normal)
                : SurfaceFlux::numerical_flux(eq, u_inner, u_inner, normal);
        for (std::size_t var = 0; var < NVARS_LOCAL; ++var) {
          surface_flux(ielem, jelem, storage_dof, var) = boundary_flux[var];
        }
      };

      if (face_id == 0) {
        const std::size_t ielem = 0;
        const std::size_t jelem = static_cast<std::size_t>(index);
        for (std::size_t jnode = 0; jnode < n_nodes; ++jnode) {
          const std::size_t boundary_dof = local_dof(0, jnode);
          write_flux(ielem, jelem, boundary_dof, face_dof(0, jnode), 0, false);
        }
      } else if (face_id == 1) {
        const std::size_t ielem = n_cells[0] - 1;
        const std::size_t jelem = static_cast<std::size_t>(index);
        for (std::size_t jnode = 0; jnode < n_nodes; ++jnode) {
          const std::size_t boundary_dof = local_dof(n_nodes - 1, jnode);
          write_flux(ielem, jelem, boundary_dof, face_dof(1, jnode), 0, true);
        }
      } else if (face_id == 2) {
        const std::size_t ielem = static_cast<std::size_t>(index);
        const std::size_t jelem = 0;
        for (std::size_t inode = 0; inode < n_nodes; ++inode) {
          const std::size_t boundary_dof = local_dof(inode, 0);
          write_flux(ielem, jelem, boundary_dof, face_dof(2, inode), 1, false);
        }
      } else {
        const std::size_t ielem = static_cast<std::size_t>(index);
        const std::size_t jelem = n_cells[1] - 1;
        for (std::size_t inode = 0; inode < n_nodes; ++inode) {
          const std::size_t boundary_dof = local_dof(inode, n_nodes - 1);
          write_flux(ielem, jelem, boundary_dof, face_dof(3, inode), 1, true);
        }
      }
    }
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
               std::size_t face_id, T time, int index = 0) const {
    using traits = equations::EquationTraits<Equations>;
    constexpr std::size_t NVARS = traits::NVARS;

    static_assert(NDIMS == 2,
                  "MixedDirichletSlipWallBC currently supports only 2D.");
    static_assert(NVARS == 4,
                  "MixedDirichletSlipWallBC currently supports 2D "
                  "compressible Euler.");

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
    auto get_boundary_coord = [&](std::size_t ielem, std::size_t jelem,
                                  std::size_t dof) {
      std::array<T, 2> coord{};
      coord[0] = element_data.node_coordinates(ielem, jelem, dof, 0);
      coord[1] = element_data.node_coordinates(ielem, jelem, dof, 1);
      return coord;
    };
    auto get_normal = [&](std::size_t ielem, std::size_t jelem,
                          std::size_t dof, std::size_t dim) {
      const T sign_jacobian =
          element_data.inverse_jacobian(ielem, jelem, dof) >= T{0} ? T{1}
                                                                    : T{-1};
      return std::array<T, 2>{
          sign_jacobian *
              element_data.contravariant_vectors(ielem, jelem, dof, dim, 0),
          sign_jacobian *
              element_data.contravariant_vectors(ielem, jelem, dof, dim, 1)};
    };

    auto write_flux = [&](std::size_t ielem, std::size_t jelem,
                          std::size_t boundary_dof, std::size_t storage_dof,
                          std::size_t dim, bool left_state_inner) {
      const auto coord = get_boundary_coord(ielem, jelem, boundary_dof);
      std::array<T, NVARS> u_inner{};
      for (std::size_t var = 0; var < NVARS; ++var) {
        u_inner[var] = u(ielem, jelem, boundary_dof, var);
      }

      const auto normal = get_normal(ielem, jelem, boundary_dof, dim);
      const auto boundary_flux = [&]() {
        if (use_dirichlet(coord, time)) {
          const auto u_outer = func(coord, time);
          return left_state_inner
                     ? SurfaceFlux::numerical_flux(eq, u_inner, u_outer, normal)
                     : SurfaceFlux::numerical_flux(eq, u_outer, u_inner, normal);
        }

        const auto u_wall = reflect_state(u_inner, normal);
        return left_state_inner
                   ? SurfaceFlux::numerical_flux(eq, u_inner, u_wall, normal)
                   : SurfaceFlux::numerical_flux(eq, u_wall, u_inner, normal);
      }();

      for (std::size_t var = 0; var < NVARS; ++var) {
        surface_flux(ielem, jelem, storage_dof, var) = boundary_flux[var];
      }
    };

    if (face_id == 0) {
      const std::size_t ielem = 0;
      const std::size_t jelem = static_cast<std::size_t>(index);
      for (std::size_t jnode = 0; jnode < n_nodes; ++jnode) {
        const std::size_t boundary_dof = local_dof(0, jnode);
        write_flux(ielem, jelem, boundary_dof, face_dof(0, jnode), 0, false);
      }
    } else if (face_id == 1) {
      const std::size_t ielem = n_cells[0] - 1;
      const std::size_t jelem = static_cast<std::size_t>(index);
      for (std::size_t jnode = 0; jnode < n_nodes; ++jnode) {
        const std::size_t boundary_dof = local_dof(n_nodes - 1, jnode);
        write_flux(ielem, jelem, boundary_dof, face_dof(1, jnode), 0, true);
      }
    } else if (face_id == 2) {
      const std::size_t ielem = static_cast<std::size_t>(index);
      const std::size_t jelem = 0;
      for (std::size_t inode = 0; inode < n_nodes; ++inode) {
        const std::size_t boundary_dof = local_dof(inode, 0);
        write_flux(ielem, jelem, boundary_dof, face_dof(2, inode), 1, false);
      }
    } else {
      const std::size_t ielem = static_cast<std::size_t>(index);
      const std::size_t jelem = n_cells[1] - 1;
      for (std::size_t inode = 0; inode < n_nodes; ++inode) {
        const std::size_t boundary_dof = local_dof(inode, n_nodes - 1);
        write_flux(ielem, jelem, boundary_dof, face_dof(3, inode), 1, true);
      }
    }
  }
};

struct SlipWallBC {
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
               std::size_t face_id, T time, int index = 0) const {
    using traits = equations::EquationTraits<Equations>;
    constexpr std::size_t NVARS = traits::NVARS;

    static_assert(NDIMS == 2, "SlipWallBC currently supports only 2D.");
    static_assert(NVARS == 4,
                  "SlipWallBC currently supports 2D compressible Euler.");

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
    auto get_normal = [&](std::size_t ielem, std::size_t jelem,
                          std::size_t dof, std::size_t dim) {
      const T sign_jacobian =
          element_data.inverse_jacobian(ielem, jelem, dof) >= T{0} ? T{1}
                                                                    : T{-1};
      return std::array<T, 2>{
          sign_jacobian *
              element_data.contravariant_vectors(ielem, jelem, dof, dim, 0),
          sign_jacobian *
              element_data.contravariant_vectors(ielem, jelem, dof, dim, 1)};
    };

    auto write_flux = [&](std::size_t ielem, std::size_t jelem,
                          std::size_t boundary_dof, std::size_t storage_dof,
                          std::size_t dim, bool left_state_inner) {
      std::array<T, NVARS> u_inner{};
      for (std::size_t var = 0; var < NVARS; ++var) {
        u_inner[var] = u(ielem, jelem, boundary_dof, var);
      }
      const auto normal = get_normal(ielem, jelem, boundary_dof, dim);
      const auto u_wall = reflect_state(u_inner, normal);
      const auto boundary_flux =
          left_state_inner
              ? SurfaceFlux::numerical_flux(eq, u_inner, u_wall, normal)
              : SurfaceFlux::numerical_flux(eq, u_wall, u_inner, normal);
      for (std::size_t var = 0; var < NVARS; ++var) {
        surface_flux(ielem, jelem, storage_dof, var) = boundary_flux[var];
      }
    };

    if (face_id == 0) {
      const std::size_t ielem = 0;
      const std::size_t jelem = static_cast<std::size_t>(index);
      for (std::size_t jnode = 0; jnode < n_nodes; ++jnode) {
        const std::size_t boundary_dof = local_dof(0, jnode);
        write_flux(ielem, jelem, boundary_dof, face_dof(0, jnode), 0, false);
      }
    } else if (face_id == 1) {
      const std::size_t ielem = n_cells[0] - 1;
      const std::size_t jelem = static_cast<std::size_t>(index);
      for (std::size_t jnode = 0; jnode < n_nodes; ++jnode) {
        const std::size_t boundary_dof = local_dof(n_nodes - 1, jnode);
        write_flux(ielem, jelem, boundary_dof, face_dof(1, jnode), 0, true);
      }
    } else if (face_id == 2) {
      const std::size_t ielem = static_cast<std::size_t>(index);
      const std::size_t jelem = 0;
      for (std::size_t inode = 0; inode < n_nodes; ++inode) {
        const std::size_t boundary_dof = local_dof(inode, 0);
        write_flux(ielem, jelem, boundary_dof, face_dof(2, inode), 1, false);
      }
    } else {
      const std::size_t ielem = static_cast<std::size_t>(index);
      const std::size_t jelem = n_cells[1] - 1;
      for (std::size_t inode = 0; inode < n_nodes; ++inode) {
        const std::size_t boundary_dof = local_dof(inode, n_nodes - 1);
        write_flux(ielem, jelem, boundary_dof, face_dof(3, inode), 1, true);
      }
    }
  }
};

template <class Func>
struct DirichletBC {
  Func func;

  explicit DirichletBC(Func func_) : func(func_) {}

  // Kokkos implementation
  template <class Equations, class SurfaceFlux, class Mesh, class T,
            std::size_t NDIMS, class ArrayU, class ElementData, class ArrayFlux>
  KOKKOS_INLINE_FUNCTION void
  apply_device(const Mesh& mesh, const Equations& eq, const ArrayU& u,
               const ElementData& element_data, ArrayFlux& surface_flux,
               std::size_t face_id, T time, int index = 0) const {
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

    if constexpr (NDIMS == 1) {
      std::size_t n_cells = mesh.get_num_cells(0);
      std::size_t n_nodes = u.extent(1);

      if (face_id == 0) {
        std::array<T, NVARS> u_boundary{};
        for (std::size_t var = 0; var < NVARS; ++var) {
          u_boundary[var] = u(0, 0, var);
        }
        std::array<T, 1> coord{mesh.get_face_coord(0)};
        std::array<T, NVARS> u_outer = get_u_outer_device(coord, time);

        auto boundary_flux =
            SurfaceFlux::numerical_flux(eq, u_outer, u_boundary);
        for (std::size_t var = 0; var < NVARS; ++var) {
          surface_flux(0, 0, var) = boundary_flux[var];
        }
      } else {
        std::array<T, NVARS> u_boundary{};
        for (std::size_t var = 0; var < NVARS; ++var) {
          u_boundary[var] = u(n_cells - 1, n_nodes - 1, var);
        }
        std::array<T, 1> coord{mesh.get_face_coord(1)};
        std::array<T, NVARS> u_outer = get_u_outer_device(coord, time);

        auto boundary_flux =
            SurfaceFlux::numerical_flux(eq, u_boundary, u_outer);
        for (std::size_t var = 0; var < NVARS; ++var) {
          surface_flux(n_cells - 1, 1, var) = boundary_flux[var];
        }
      }
    } else if constexpr (NDIMS == 2) {
      auto get_boundary_coord = [&](std::size_t ielem, std::size_t jelem,
                                    std::size_t dof) {
        std::array<T, 2> coord{};
        coord[0] = element_data.node_coordinates(ielem, jelem, dof, 0);
        coord[1] = element_data.node_coordinates(ielem, jelem, dof, 1);
        return coord;
      };

      auto get_normal = [&](std::size_t ielem, std::size_t jelem,
                            std::size_t dof, std::size_t dim) {
        const T sign_jacobian =
            element_data.inverse_jacobian(ielem, jelem, dof) >= T{0} ? T{1}
                                                                      : T{-1};
        return std::array<T, 2>{
            sign_jacobian *
                element_data.contravariant_vectors(ielem, jelem, dof, dim, 0),
            sign_jacobian *
                element_data.contravariant_vectors(ielem, jelem, dof, dim, 1)};
      };

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

      if (face_id == 0) {
        const std::size_t ielem = 0;
        const std::size_t jelem = static_cast<std::size_t>(index);
        for (std::size_t jnode = 0; jnode < n_nodes; ++jnode) {
          const std::size_t boundary_dof = local_dof(0, jnode);
          const auto coord = get_boundary_coord(ielem, jelem, boundary_dof);
          const auto u_outer = get_u_outer_device(coord, time);
          const auto normal = get_normal(ielem, jelem, boundary_dof, 0);

          std::array<T, NVARS> u_boundary{};
          for (std::size_t var = 0; var < NVARS; ++var) {
            u_boundary[var] = u(ielem, jelem, boundary_dof, var);
          }

          const auto boundary_flux =
              SurfaceFlux::numerical_flux(eq, u_outer, u_boundary, normal);
          for (std::size_t var = 0; var < NVARS; ++var) {
            surface_flux(ielem, jelem, face_dof(0, jnode), var) =
                boundary_flux[var];
          }
        }
      } else if (face_id == 1) {
        const std::size_t ielem = n_cells[0] - 1;
        const std::size_t jelem = static_cast<std::size_t>(index);
        for (std::size_t jnode = 0; jnode < n_nodes; ++jnode) {
          const std::size_t boundary_dof = local_dof(n_nodes - 1, jnode);
          const auto coord = get_boundary_coord(ielem, jelem, boundary_dof);
          const auto u_outer = get_u_outer_device(coord, time);
          const auto normal = get_normal(ielem, jelem, boundary_dof, 0);

          std::array<T, NVARS> u_boundary{};
          for (std::size_t var = 0; var < NVARS; ++var) {
            u_boundary[var] = u(ielem, jelem, boundary_dof, var);
          }

          const auto boundary_flux =
              SurfaceFlux::numerical_flux(eq, u_boundary, u_outer, normal);
          for (std::size_t var = 0; var < NVARS; ++var) {
            surface_flux(ielem, jelem, face_dof(1, jnode), var) =
                boundary_flux[var];
          }
        }
      } else if (face_id == 2) {
        const std::size_t ielem = static_cast<std::size_t>(index);
        const std::size_t jelem = 0;
        for (std::size_t inode = 0; inode < n_nodes; ++inode) {
          const std::size_t boundary_dof = local_dof(inode, 0);
          const auto coord = get_boundary_coord(ielem, jelem, boundary_dof);
          const auto u_outer = get_u_outer_device(coord, time);
          const auto normal = get_normal(ielem, jelem, boundary_dof, 1);

          std::array<T, NVARS> u_boundary{};
          for (std::size_t var = 0; var < NVARS; ++var) {
            u_boundary[var] = u(ielem, jelem, boundary_dof, var);
          }

          const auto boundary_flux =
              SurfaceFlux::numerical_flux(eq, u_outer, u_boundary, normal);
          for (std::size_t var = 0; var < NVARS; ++var) {
            surface_flux(ielem, jelem, face_dof(2, inode), var) =
                boundary_flux[var];
          }
        }
      } else {
        const std::size_t ielem = static_cast<std::size_t>(index);
        const std::size_t jelem = n_cells[1] - 1;
        for (std::size_t inode = 0; inode < n_nodes; ++inode) {
          const std::size_t boundary_dof = local_dof(inode, n_nodes - 1);
          const auto coord = get_boundary_coord(ielem, jelem, boundary_dof);
          const auto u_outer = get_u_outer_device(coord, time);
          const auto normal = get_normal(ielem, jelem, boundary_dof, 1);

          std::array<T, NVARS> u_boundary{};
          for (std::size_t var = 0; var < NVARS; ++var) {
            u_boundary[var] = u(ielem, jelem, boundary_dof, var);
          }

          const auto boundary_flux =
              SurfaceFlux::numerical_flux(eq, u_boundary, u_outer, normal);
          for (std::size_t var = 0; var < NVARS; ++var) {
            surface_flux(ielem, jelem, face_dof(3, inode), var) =
                boundary_flux[var];
          }
        }
      }
    }
  }
};
} // namespace DGSEM
