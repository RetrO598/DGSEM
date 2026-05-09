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

template <class T, std::size_t NDIMS>
struct NodeVectorFieldInitializer;

template <class Equations, class = void>
struct ParabolicEquationInfo {
  constexpr static bool enabled = false;
  constexpr static std::size_t ngrad_vars = 0;
};

template <class Equations>
struct ParabolicEquationInfo<Equations,
                             std::void_t<decltype(Equations::NGRAD_VARS)>> {
  constexpr static bool enabled = true;
  constexpr static std::size_t ngrad_vars = Equations::NGRAD_VARS;
};

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

template <class T, std::size_t NVARS>
struct SolutionInitializer<T, NVARS, 3> {
  using DataArray = solution_type_traits<T, 3>::DataArray;
  using DataArrayHost = solution_type_traits<T, 3>::DataArrayHost;

  inline constexpr static void
  initialize_u(const std::array<std::size_t, 3>& n_cells, std::size_t ndofs,
               DataArray& u_device) {
    Kokkos::realloc(u_device, n_cells[0], n_cells[1], n_cells[2], ndofs, NVARS);
    Kokkos::deep_copy(u_device, T{0.0});
  }

  inline constexpr static void
  initialize_du(const std::array<std::size_t, 3>& n_cells, std::size_t ndofs,
                DataArray& du_device) {
    Kokkos::realloc(du_device, n_cells[0], n_cells[1], n_cells[2], ndofs,
                    NVARS);
    Kokkos::deep_copy(du_device, T{0.0});
  }

  inline constexpr static void
  initialize_surface_flux_value(const std::array<std::size_t, 3>& n_cells,
                                std::size_t nnodes, DataArray& surface_device) {
    Kokkos::realloc(surface_device, n_cells[0], n_cells[1], n_cells[2],
                    6 * nnodes * nnodes, NVARS);
    Kokkos::deep_copy(surface_device, T{0.0});
  }
};

template <class T>
struct NodeVectorFieldInitializer<T, 1> {
  using DataArray = typename node_vector_field_type_traits<T, 1>::DataArray;

  inline constexpr static void
  initialize(const std::array<std::size_t, 1>& n_cells, std::size_t nnodes,
             std::size_t nvars, std::size_t ncomponents,
             DataArray& data_device) {
    Kokkos::realloc(data_device, n_cells[0], nnodes, nvars, ncomponents);
    Kokkos::deep_copy(data_device, T{0.0});
  }
};

template <class T>
struct NodeVectorFieldInitializer<T, 2> {
  using DataArray = typename node_vector_field_type_traits<T, 2>::DataArray;

  inline constexpr static void
  initialize(const std::array<std::size_t, 2>& n_cells, std::size_t ndofs,
             std::size_t nvars, std::size_t ncomponents,
             DataArray& data_device) {
    Kokkos::realloc(data_device, n_cells[0], n_cells[1], ndofs, nvars,
                    ncomponents);
    Kokkos::deep_copy(data_device, T{0.0});
  }
};

template <class T>
struct NodeVectorFieldInitializer<T, 3> {
  using DataArray = typename node_vector_field_type_traits<T, 3>::DataArray;

  inline constexpr static void
  initialize(const std::array<std::size_t, 3>& n_cells, std::size_t ndofs,
             std::size_t nvars, std::size_t ncomponents,
             DataArray& data_device) {
    Kokkos::realloc(data_device, n_cells[0], n_cells[1], n_cells[2], ndofs,
                    nvars, ncomponents);
    Kokkos::deep_copy(data_device, T{0.0});
  }
};
} // namespace detail

template <class Mesh, class Basis, class Equations>
struct Solution {
  using traits = equations::EquationTraits<Equations>;
  using value_type = typename traits::value_type;
  constexpr static std::size_t NDIMS = traits::NDIMS;
  constexpr static std::size_t NVARS = traits::NVARS;
  constexpr static bool HasParabolicTerms =
      detail::ParabolicEquationInfo<Equations>::enabled;
  constexpr static std::size_t NGRAD_VARS =
      detail::ParabolicEquationInfo<Equations>::ngrad_vars;

  using DataArray = solution_type_traits<value_type, NDIMS>::DataArray;
  using DataArrayHost = solution_type_traits<value_type, NDIMS>::DataArrayHost;
  using VectorFieldArray =
      node_vector_field_type_traits<value_type, NDIMS>::DataArray;

  Solution(const Mesh& mesh) {
    auto n_cells = mesh.get_num_cells();
    constexpr std::size_t ndofs =
        (NDIMS == 1)
            ? Basis::NNodes
            : ((NDIMS == 2) ? Basis::NNodes * Basis::NNodes
                            : Basis::NNodes * Basis::NNodes * Basis::NNodes);
    detail::SolutionInitializer<value_type, NVARS, NDIMS>::initialize_u(
        n_cells, ndofs, u_device);

    detail::SolutionInitializer<value_type, NVARS, NDIMS>::initialize_du(
        n_cells, ndofs, du_device);

    detail::SolutionInitializer<value_type, NVARS, NDIMS>::
        initialize_surface_flux_value(n_cells, Basis::NNodes,
                                      surface_flux_value_device);

    if constexpr (HasParabolicTerms) {
      detail::NodeVectorFieldInitializer<value_type, NDIMS>::initialize(
          n_cells, ndofs, NGRAD_VARS, NDIMS, gradient_device);
      detail::NodeVectorFieldInitializer<value_type, NDIMS>::initialize(
          n_cells, ndofs, NVARS, NDIMS, viscous_flux_device);
      detail::SolutionInitializer<value_type, NGRAD_VARS, NDIMS>::
          initialize_surface_flux_value(n_cells, Basis::NNodes,
                                        gradient_surface_flux_device);
      detail::SolutionInitializer<value_type, NVARS, NDIMS>::
          initialize_surface_flux_value(n_cells, Basis::NNodes,
                                        viscous_surface_flux_value_device);
    }
  }

  Solution() = default;

  Solution clone_shape() const {
    Solution tmp;

    DGSEM::utils::clone_view_shape(tmp.u_device, u_device);
    DGSEM::utils::clone_view_shape(tmp.du_device, du_device);
    DGSEM::utils::clone_view_shape(tmp.surface_flux_value_device,
                                   surface_flux_value_device);

    if constexpr (HasParabolicTerms) {
      DGSEM::utils::clone_view_shape(tmp.gradient_device, gradient_device);
      DGSEM::utils::clone_view_shape(tmp.viscous_flux_device,
                                     viscous_flux_device);
      DGSEM::utils::clone_view_shape(tmp.gradient_surface_flux_device,
                                     gradient_surface_flux_device);
      DGSEM::utils::clone_view_shape(tmp.viscous_surface_flux_value_device,
                                     viscous_surface_flux_value_device);
    }

    return tmp;
  }

  DataArray& state() { return u_device; }
  const DataArray& state() const { return u_device; }

  DataArray& rhs() { return du_device; }
  const DataArray& rhs() const { return du_device; }

  DataArray& surface_flux() { return surface_flux_value_device; }
  const DataArray& surface_flux() const { return surface_flux_value_device; }

  VectorFieldArray& gradient()
    requires(HasParabolicTerms)
  {
    return gradient_device;
  }

  const VectorFieldArray& gradient() const
    requires(HasParabolicTerms)
  {
    return gradient_device;
  }

  VectorFieldArray& viscous_flux()
    requires(HasParabolicTerms)
  {
    return viscous_flux_device;
  }

  const VectorFieldArray& viscous_flux() const
    requires(HasParabolicTerms)
  {
    return viscous_flux_device;
  }

  DataArray& gradient_surface_flux()
    requires(HasParabolicTerms)
  {
    return gradient_surface_flux_device;
  }

  const DataArray& gradient_surface_flux() const
    requires(HasParabolicTerms)
  {
    return gradient_surface_flux_device;
  }

  DataArray& viscous_surface_flux()
    requires(HasParabolicTerms)
  {
    return viscous_surface_flux_value_device;
  }

  const DataArray& viscous_surface_flux() const
    requires(HasParabolicTerms)
  {
    return viscous_surface_flux_value_device;
  }

  DataArray u_device;
  DataArray du_device;
  DataArray surface_flux_value_device;
  VectorFieldArray gradient_device;
  VectorFieldArray viscous_flux_device;
  DataArray gradient_surface_flux_device;
  DataArray viscous_surface_flux_value_device;
};
} // namespace DGSEM
