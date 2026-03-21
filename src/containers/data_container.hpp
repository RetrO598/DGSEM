#pragma once

#include "base/global_def.hpp"
#include "utils/kokkos_helper.hpp"
#include <array>
#include <cstddef>
#include <equations/equations.hpp>
#include <functional>
#include <numeric>
#include <xtensor.hpp>
#include <xtensor/core/xtensor_forward.hpp>
namespace DGSEM {
template <class T, std::size_t NDIMS>
struct StructuredElementContainer {
  using ndarray = xt::xarray<T>;
  using index_array = xt::xarray<std::size_t>;

  using DataArray = solution_type_traits<T, NDIMS>::DataArray;
  using DataArrayHost = solution_type_traits<T, NDIMS>::DataArrayHost;
  using IndexArray = index_type_traits<NDIMS>::IndexArray;
  using Matrix = jacobian_type_traits<T, NDIMS>::JacobianMatrix;

  std::array<std::size_t, NDIMS> nelements;
  ndarray node_coordinates;
  index_array left_neighbors;
  ndarray jacobian_matrix;
  ndarray contravariant_vectors;
  ndarray inverse_jacobian;

  DataArray node_coordinates_kokkos;
  IndexArray left_neighbors_kokkos;
  Matrix jacobian_matrix_kokkos;
  Matrix contravariant_vectors_kokkos;
  DataArray inverse_jacobian_kokkos;

  void sync_to_device() {
    copy_xtensor_to_kokkos(node_coordinates_kokkos, node_coordinates);
    copy_xtensor_to_kokkos(left_neighbors_kokkos, left_neighbors);
    copy_xtensor_to_kokkos(jacobian_matrix_kokkos, jacobian_matrix);
    copy_xtensor_to_kokkos(contravariant_vectors_kokkos, contravariant_vectors);
    copy_xtensor_to_kokkos(inverse_jacobian_kokkos, inverse_jacobian);
  }

  void check_data() {
    compare_view_xtensor(node_coordinates_kokkos, node_coordinates);
    compare_view_xtensor(left_neighbors_kokkos, left_neighbors);
    compare_view_xtensor(jacobian_matrix_kokkos, jacobian_matrix);
    compare_view_xtensor(contravariant_vectors_kokkos, contravariant_vectors);
    compare_view_xtensor(inverse_jacobian_kokkos, inverse_jacobian);
  }
};

namespace detail {
template <class T, class Basis, class Mapping, std::size_t NDIMS>
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
    container.inverse_jacobian.resize({tot_elems, Basis::NNodes, 1});
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
      inverse_jacobian(ielem, i, 0) = 1.0 / jacobian(ielem, i, 0, 0);
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

template <class T, class Basis, class Mapping, std::size_t NDIMS>
struct StructuredElementInitializer {
  using ndarray = StructuredElementContainer<T, NDIMS>::ndarray;
  using index_array = StructuredElementContainer<T, NDIMS>::index_array;

  StructuredElementInitializer(Mapping mapping_,
                               std::array<bool, NDIMS> periodic_)
      : mapping(mapping_), periodic(periodic_) {}

  constexpr void
  init_elements(std::array<std::size_t, NDIMS> n_cells,
                StructuredElementContainer<T, NDIMS> &container) {
    container.nelements = n_cells;
    std::size_t total_elements = std::accumulate(
        n_cells.begin(), n_cells.end(), 1.0, std::multiplies<std::size_t>());
    detail::StructuredContainerInitializer<T, Basis, Mapping, NDIMS>::resize(
        total_elements, container);

    for (std::size_t i = 0; i < total_elements; ++i) {
      detail::StructuredContainerInitializer<T, Basis, Mapping, NDIMS>::
          calc_node_coordinates(total_elements, i, container.node_coordinates,
                                mapping);
      detail::StructuredContainerInitializer<T, Basis, Mapping, NDIMS>::
          calc_jacobian_matrix(i, container.jacobian_matrix,
                               container.node_coordinates);
      detail::StructuredContainerInitializer<T, Basis, Mapping, NDIMS>::
          calc_inverse_jacobian(i, container.inverse_jacobian,
                                container.jacobian_matrix);
    }

    detail::StructuredContainerInitializer<T, Basis, Mapping, NDIMS>::
        initialize_left_neighbor(total_elements, container.left_neighbors,
                                 periodic);
  }

  Mapping mapping;
  std::array<bool, NDIMS> periodic;
};
} // namespace DGSEM
