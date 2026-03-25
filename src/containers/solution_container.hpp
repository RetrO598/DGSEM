#pragma once

#include <Kokkos_Core.hpp>
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

  KOKKOS_INLINE_FUNCTION constexpr static void
  initialize_u(std::size_t total_elements, std::size_t nnodes,
               DataArray &u_device) {
    Kokkos::realloc(u_device, total_elements, nnodes, NVARS);
    Kokkos::deep_copy(u_device, 0.0);
  }

  KOKKOS_INLINE_FUNCTION constexpr static void
  initialize_du(std::size_t total_elements, std::size_t nnodes,
                DataArray &du_device) {
    Kokkos::realloc(du_device, total_elements, nnodes, NVARS);
    Kokkos::deep_copy(du_device, 0.0);
  }

  KOKKOS_INLINE_FUNCTION constexpr static void
  initialize_surface_flux_value(std::size_t total_elements, std::size_t nnodes,
                                DataArray &surface_device) {
    Kokkos::realloc(surface_device, total_elements, 2, NVARS);
    Kokkos::deep_copy(surface_device, 0.0);
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

  Solution(const Mesh &mesh) {
    std::size_t total_elements = mesh.get_nelem();
    detail::SolutionInitializer<value_type, NVARS, NDIMS>::initialize_u(
        total_elements, Basis::NNodes, u_device);

    detail::SolutionInitializer<value_type, NVARS, NDIMS>::initialize_du(
        total_elements, Basis::NNodes, du_device);

    detail::SolutionInitializer<value_type, NVARS, NDIMS>::
        initialize_surface_flux_value(total_elements, Basis::NNodes,
                                      surface_flux_value_device);
  }

  Solution() = default;

  Solution clone_shape() const {
    Solution tmp;

    clone_view_shape(tmp.u_device, u_device);
    clone_view_shape(tmp.du_device, du_device);
    clone_view_shape(tmp.surface_flux_value_device, surface_flux_value_device);

    return tmp;
  }

  DataArray u_device;
  DataArray du_device;
  DataArray surface_flux_value_device;
};
} // namespace DGSEM