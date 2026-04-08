#pragma once

#include <Kokkos_Core.hpp>
#include <array>
#include <base/base.hpp>
#include <cstddef>
#include <equations/equations.hpp>
#include <utils/utils.hpp>

namespace DGSEM {

namespace detail {
template <class T, std::size_t NVARS, std::size_t NDIMS>
struct SolutionInitializer;

template <class T, std::size_t NVARS>
struct SolutionInitializer<T, NVARS, 1> {
  using DataArray = solution_type_traits<T, 1>::DataArray;
  using DataArrayHost = solution_type_traits<T, 1>::DataArrayHost;

  inline constexpr static void
  initialize_u(const std::array<std::size_t, 1>& n_cells, std::size_t nnodes,
               DataArray& u_device) {
    Kokkos::realloc(u_device, n_cells[0], nnodes, NVARS);
    Kokkos::deep_copy(u_device, T{0.0});
  }

  inline constexpr static void
  initialize_du(const std::array<std::size_t, 1>& n_cells, std::size_t nnodes,
                DataArray& du_device) {
    Kokkos::realloc(du_device, n_cells[0], nnodes, NVARS);
    Kokkos::deep_copy(du_device, T{});
  }

  inline constexpr static void
  initialize_surface_flux_value(const std::array<std::size_t, 1>& n_cells,
                                std::size_t nnodes, DataArray& surface_device) {
    Kokkos::realloc(surface_device, n_cells[0], 2, NVARS);
    Kokkos::deep_copy(surface_device, T{0.0});
  }
};

template <class T, std::size_t NVARS>
struct SolutionInitializer<T, NVARS, 2> {
  using DataArray = solution_type_traits<T, 2>::DataArray;
  using DataArrayHost = solution_type_traits<T, 2>::DataArrayHost;

  inline constexpr static void
  initialize_u(const std::array<std::size_t, 2>& n_cells, std::size_t ndofs,
               DataArray& u_device) {
    Kokkos::realloc(u_device, n_cells[0], n_cells[1], ndofs, NVARS);
    Kokkos::deep_copy(u_device, T{0.0});
  }

  inline constexpr static void
  initialize_du(const std::array<std::size_t, 2>& n_cells, std::size_t ndofs,
                DataArray& du_device) {
    Kokkos::realloc(du_device, n_cells[0], n_cells[1], ndofs, NVARS);
    Kokkos::deep_copy(du_device, T{0.0});
  }

  inline constexpr static void
  initialize_surface_flux_value(const std::array<std::size_t, 2>& n_cells,
                                std::size_t nnodes, DataArray& surface_device) {
    Kokkos::realloc(surface_device, n_cells[0], n_cells[1], 4 * nnodes, NVARS);
    Kokkos::deep_copy(surface_device, T{0.0});
  }
};
} // namespace detail

template <class Mesh, class Basis, class Equations>
struct Solution {
  using traits = equations::EquationTraits<Equations>;
  using value_type = typename traits::value_type;
  constexpr static std::size_t NDIMS = traits::NDIMS;
  constexpr static std::size_t NVARS = traits::NVARS;

  using DataArray = solution_type_traits<value_type, NDIMS>::DataArray;
  using DataArrayHost = solution_type_traits<value_type, NDIMS>::DataArrayHost;

  Solution(const Mesh& mesh) {
    auto n_cells = mesh.get_num_cells();
    constexpr std::size_t ndofs =
        (NDIMS == 1) ? Basis::NNodes : Basis::NNodes * Basis::NNodes;
    detail::SolutionInitializer<value_type, NVARS, NDIMS>::initialize_u(
        n_cells, ndofs, u_device);

    detail::SolutionInitializer<value_type, NVARS, NDIMS>::initialize_du(
        n_cells, ndofs, du_device);

    detail::SolutionInitializer<value_type, NVARS, NDIMS>::
        initialize_surface_flux_value(n_cells, Basis::NNodes,
                                      surface_flux_value_device);
  }

  Solution() = default;

  Solution clone_shape() const {
    Solution tmp;

    DGSEM::utils::clone_view_shape(tmp.u_device, u_device);
    DGSEM::utils::clone_view_shape(tmp.du_device, du_device);
    DGSEM::utils::clone_view_shape(tmp.surface_flux_value_device,
                                   surface_flux_value_device);

    return tmp;
  }

  DataArray u_device;
  DataArray du_device;
  DataArray surface_flux_value_device;
};
} // namespace DGSEM
