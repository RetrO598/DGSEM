#pragma once

#include <array>
#include <cstddef>
#include <equations/equations.hpp>
#include <functional>
#include <numeric>
#include <xtensor.hpp>
#include <xtensor/core/xtensor_forward.hpp>
namespace DGSEM {
template <class T, std::size_t NDIM>
struct StructuredElementContainer {
  using ndarray = xt::xarray<T>;
  using index_array = xt::xarray<std::size_t>;

  std::array<std::size_t, NDIM> nelements;
  ndarray node_coordinates;
  index_array left_neighbors;
  ndarray jacobian_matrix;
  ndarray contravariant_vectors;
  ndarray inverse_jacobian;
};

namespace detail {
template <class T, class Basis, class Mapping, std::size_t NDIM>
struct StructuredContainerInitializer;

template <class T, class Basis, class Mapping>
struct StructuredContainerInitializer<T, Basis, Mapping, 1> {
  using ndarray = StructuredElementContainer<T, 1>::ndarray;
  using index_array = StructuredElementContainer<T, 1>::index_array;

  inline constexpr static void
  resize(std::size_t tot_elems, StructuredElementContainer<T, 1> &container) {

    container.node_coordinates.resize({tot_elems, Basis::NNodes, 1});
    container.left_neighbors.resize({tot_elems, 1});
    container.jacobian_matrix.resize({
        tot_elems,
        Basis::NNodes,
        1,
        1,
    });
    container.contravariant_vectors.resize({tot_elems, Basis::NNodes, 1, 1});
    container.inverse_jacobian.resize({tot_elems, Basis::NNodes});
  }

  inline constexpr static void calc_node_coordinates(std::size_t nelems,
                                                     std::size_t ielem,
                                                     ndarray &coordinates,
                                                     const Mapping &mapping) {
    T dx = 2.0 / nelems;
    T cell_x_offset = -1.0 + ielem * dx + 0.5 * dx;

    for (std::size_t i = 0; i < Basis::NNodes; ++i) {
      coordinates(ielem, i, 0) =
          mapping.eval(cell_x_offset + 0.5 * dx * Basis::nodes[i]);
    }
  }

  inline constexpr static void
  calc_jacobian_matrix(std::size_t ielem, ndarray &jacobian,
                       const ndarray &coordinates) {
    for (std::size_t i = 0; i < Basis::NNodes; ++i) {
      T tmp = 0.0;
      for (std::size_t j = 0; j < Basis::NNodes; ++j) {
        tmp += Basis::derivative_matrix(i, j) * coordinates(ielem, j, 0);
      }
      jacobian(ielem, i, 0, 0) = tmp;
    }
  }

  inline constexpr static void calc_inverse_jacobian(std::size_t ielem,
                                                     ndarray &inverse_jacobian,
                                                     const ndarray &jacobian) {
    for (std::size_t i = 0; i < Basis::NNodes; ++i) {
      inverse_jacobian(ielem, i) = 1.0 / jacobian(ielem, i, 0, 0);
    }
  }

  inline constexpr static void
  initialize_left_neighbor(std::size_t nelems, index_array &left_neighbors,
                           std::array<bool, 1> periodic) {
    for (std::size_t i = 1; i < nelems; ++i) {
      left_neighbors(i, 0) = i - 1;
    }

    if (periodic[0] == true) {
      left_neighbors(0, 0) = nelems - 1;
    } else {
      left_neighbors(0, 0) = static_cast<std::size_t>(-1);
    }
  }
};
} // namespace detail

template <class T, class Basis, class Mapping, std::size_t NDIM>
struct StructuredElementInitializer {
  using ndarray = StructuredElementContainer<T, NDIM>::ndarray;
  using index_array = StructuredElementContainer<T, NDIM>::index_array;

  StructuredElementInitializer(Mapping mapping_,
                               std::array<bool, NDIM> periodic_)
      : mapping(mapping_), periodic(periodic_) {}

  constexpr void init_elements(std::array<std::size_t, NDIM> n_cells,
                               StructuredElementContainer<T, NDIM> &container) {
    container.nelements = n_cells;
    std::size_t total_elements = std::accumulate(
        n_cells.begin(), n_cells.end(), 1.0, std::multiplies<std::size_t>());
    detail::StructuredContainerInitializer<T, Basis, Mapping, NDIM>::resize(
        total_elements, container);

    for (std::size_t i = 0; i < total_elements; ++i) {
      detail::StructuredContainerInitializer<T, Basis, Mapping, NDIM>::
          calc_node_coordinates(total_elements, i, container.node_coordinates,
                                mapping);
      detail::StructuredContainerInitializer<T, Basis, Mapping, NDIM>::
          calc_jacobian_matrix(i, container.jacobian_matrix,
                               container.node_coordinates);
      detail::StructuredContainerInitializer<T, Basis, Mapping, NDIM>::
          calc_inverse_jacobian(i, container.inverse_jacobian,
                                container.jacobian_matrix);
    }

    detail::StructuredContainerInitializer<T, Basis, Mapping, NDIM>::
        initialize_left_neighbor(total_elements, container.left_neighbors,
                                 periodic);
  }

  Mapping mapping;
  std::array<bool, NDIM> periodic;
};

namespace detail {
template <class T, std::size_t NVARS, std::size_t NDIMS>
struct SolutionInitializer;

template <class T, std::size_t NVARS>
struct SolutionInitializer<T, NVARS, 1> {

  using ndarray = xt::xarray<T>;

  inline constexpr static void initialize_u(std::size_t total_elements,
                                            std::size_t nnodes, ndarray &u) {
    u.resize({total_elements, nnodes, NVARS});
    u.fill(0.0);
  }

  inline constexpr static void initialize_du(std::size_t total_elements,
                                             std::size_t nnodes, ndarray &du) {
    du.resize({total_elements, nnodes, NVARS});
    du.fill(0.0);
  }

  inline constexpr static void
  initialize_surface_flux_value(std::size_t total_elements, std::size_t nnodes,
                                ndarray &surface_flux_value) {
    surface_flux_value.resize({total_elements, 2, NVARS});
    surface_flux_value.fill(0.0);
  }
};
} // namespace detail

template <class Mesh, class Basis, class Equations>
struct Solution {
  using traits = equations::EquationTraits<Equations>;
  using value_type = typename traits::value_type;
  constexpr static std::size_t NDIMS = traits::NDIMS;
  constexpr static std::size_t NVARS = traits::NVARS;

  using ndarray = xt::xarray<value_type>;

  Solution(const Mesh &mesh) {
    std::size_t total_elements = mesh.get_nelem();
    detail::SolutionInitializer<value_type, NVARS, NDIMS>::initialize_u(
        total_elements, Basis::NNodes, u);

    detail::SolutionInitializer<value_type, NVARS, NDIMS>::initialize_du(
        total_elements, Basis::NNodes, du);

    detail::SolutionInitializer<value_type, NVARS, NDIMS>::
        initialize_surface_flux_value(total_elements, Basis::NNodes,
                                      surface_flux_value);
  }

  Solution() = default;

  Solution clone_shape() const {
    Solution tmp;
    tmp.u.resize(u.shape());
    tmp.du.resize(du.shape());
    tmp.surface_flux_value.resize(surface_flux_value.shape());
    tmp.u.fill(0.0);
    tmp.du.fill(0.0);
    tmp.surface_flux_value.fill(0.0);
    return tmp;
  }

  ndarray u;
  ndarray du;
  ndarray surface_flux_value;
};
} // namespace DGSEM
