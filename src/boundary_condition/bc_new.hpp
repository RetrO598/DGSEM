#pragma once

#include <array>
#include <cstddef>
#include <equations/equations.hpp>
#include <tuple>
#include <type_traits>
#include <xtensor/core/xtensor_forward.hpp>
namespace DGSEM {
template <class... FaceBCs>
struct BoundarySet {
  static constexpr std::size_t NFACES = sizeof...(FaceBCs);

  std::tuple<FaceBCs...> faces_;

  BoundarySet(FaceBCs... bcs) : faces_(std::move(bcs)...) {}

  template <std::size_t I = 0, typename F>
  void tuple_switch(std::size_t idx, F &&f) const {
    if constexpr (I < NFACES) {
      if (I == idx) {
        f(std::get<I>(faces_));
      } else {
        tuple_switch<I + 1>(idx, std::forward<F>(f));
      }
    } else {
      throw std::out_of_range("BoundarySet::apply: face_id out of range");
    }
  }

  template <class Equations, class SurfaceFlux, class Mesh, class T,
            std::size_t NDIMS>
  void apply(const Mesh &mesh, std::size_t NNodes, const Equations &eq,
             const xt::xarray<T> &u, xt::xarray<T> &surface_flux,
             std::size_t face_id, T time) const {
    if constexpr (NFACES > 0) {
      tuple_switch(face_id, [&](const auto &bc) {
        bc.template apply<Equations, SurfaceFlux, Mesh, T, NDIMS>(
            mesh, NNodes, eq, u, surface_flux, face_id, time);
      });
    }
  }
};

struct PeriodicBC {
  template <class Equations, class SurfaceFlux, class Mesh, class T,
            std::size_t NDIMS>
  void apply(const Mesh &mesh, std::size_t NNodes, const Equations &eq,
             const xt::xarray<T> &u, xt::xarray<T> &surface_flux,
             std::size_t face_id, T time) const {
    return;
  }
};

template <class Func>
struct DirichletBC {
  Func func;

  explicit DirichletBC(Func func_) : func(func_) {}

  template <class Equations, class SurfaceFlux, class Mesh, class T,
            std::size_t NDIMS>
  void apply(const Mesh &mesh, std::size_t NNodes, const Equations &eq,
             const xt::xarray<T> &u, xt::xarray<T> &surface_flux,
             std::size_t face_id, T time) const
    requires(NDIMS == 1)
  {
    using traits = equations::EquationTraits<Equations>;
    using value_type = typename traits::value_type;
    constexpr static std::size_t NVARS = traits::NVARS;

    if constexpr (std::is_invocable_v<Func, const std::array<T, NDIMS> &, T>) {
      if (face_id == 0) {
        std::array<T, NVARS> u_outer{};
        std::array<T, NVARS> u_boundary{};
        for (std::size_t var = 0; var < NVARS; ++var) {
          u_boundary[var] = u(0, 0, var);
        }
        std::array<T, NDIMS> coord{};
        coord[0] = mesh.get_face_coord(face_id);
        u_outer = func(coord, time);

        std::array<T, NVARS> boundary_flux =
            SurfaceFlux::numerical_flux(eq, u_outer, u_boundary);

        for (std::size_t var = 0; var < NVARS; ++var) {
          surface_flux(0, 0, var) = boundary_flux[var];
        }
      } else if (face_id == 1) {
        std::array<T, NVARS> u_outer{};
        std::array<T, NVARS> u_boundary{};
        std::size_t n_cells = mesh.get_num_cells(0);
        for (std::size_t var = 0; var < NVARS; ++var) {
          u_boundary[var] = u(n_cells - 1, NNodes - 1, var);
        }

        std::array<T, NDIMS> coord{};
        coord[0] = mesh.get_face_coord(face_id);

        u_outer = func(coord, time);

        std::array<T, NVARS> boundary_flux =
            SurfaceFlux::numerical_flux(eq, u_boundary, u_outer);

        for (std::size_t var = 0; var < NVARS; ++var) {
          surface_flux(n_cells - 1, 1, var) = boundary_flux[var];
        }
      }
    } else {
      if (face_id == 0) {
        std::array<T, NVARS> u_outer{};
        std::array<T, NVARS> u_boundary{};
        for (std::size_t var = 0; var < NVARS; ++var) {
          u_boundary[var] = u(0, 0, var);
          u_outer[var] = func[var];
        }

        std::array<T, NVARS> boundary_flux =
            SurfaceFlux::numerical_flux(eq, u_outer, u_boundary);

        for (std::size_t var = 0; var < NVARS; ++var) {
          surface_flux(0, 0, var) = boundary_flux[var];
        }
      } else if (face_id == 1) {
        std::array<T, NVARS> u_outer{};
        std::array<T, NVARS> u_boundary{};
        std::size_t n_cells = mesh.get_num_cells(face_id - 1);
        for (std::size_t var = 0; var < NVARS; ++var) {
          u_boundary[var] = u(n_cells - 1, NNodes - 1, var);
          u_outer[var] = func[var];
        }

        std::array<T, NVARS> boundary_flux =
            SurfaceFlux::numerical_flux(eq, u_boundary, u_outer);

        for (std::size_t var = 0; var < NVARS; ++var) {
          surface_flux(n_cells - 1, 1, var) = boundary_flux[var];
        }
      }
    }
  }
};
} // namespace DGSEM