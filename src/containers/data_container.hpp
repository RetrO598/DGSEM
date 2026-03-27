#pragma once

#include <Kokkos_Core.hpp>
#include <Kokkos_Macros.hpp>
#include <array>
#include <base/base.hpp>
#include <cstddef>
#include <equations/equations.hpp>
#include <iostream>

namespace DGSEM {
template <class T, std::size_t NDIMS>
struct StructuredElementContainer {
  using DataArray = solution_type_traits<T, NDIMS>::DataArray;
  using DataArrayHost = solution_type_traits<T, NDIMS>::DataArrayHost;
  using IndexArray = index_type_traits<NDIMS>::IndexArray;
  using IndexArrayHost = index_type_traits<NDIMS>::IndexArrayHost;
  using Matrix = jacobian_type_traits<T, NDIMS>::JacobianMatrix;
  using MatrixHost = jacobian_type_traits<T, NDIMS>::JacobianMatrixHost;

  std::array<std::size_t, NDIMS> nelements;
  DataArrayHost node_coordinates;
  IndexArrayHost left_neighbors;
  MatrixHost jacobian_matrix;
  MatrixHost contravariant_vectors;
  DataArrayHost inverse_jacobian;

  DataArray node_coordinates_device;
  IndexArray left_neighbors_device;
  Matrix jacobian_matrix_device;
  Matrix contravariant_vectors_device;
  DataArray inverse_jacobian_device;

  void sync_to_device() {
    Kokkos::deep_copy(node_coordinates_device, node_coordinates);
    Kokkos::deep_copy(left_neighbors_device, left_neighbors);
    Kokkos::deep_copy(jacobian_matrix_device, jacobian_matrix);
    Kokkos::deep_copy(contravariant_vectors_device, contravariant_vectors);
    Kokkos::deep_copy(inverse_jacobian_device, inverse_jacobian);
  }
};

namespace detail {
template <class T, class Basis, class Mapping, std::size_t NDIMS>
struct StructuredContainerInitializer;

template <class T, class Basis, class Mapping>
struct StructuredContainerInitializer<T, Basis, Mapping, 1> {
  using DataArray = solution_type_traits<T, 1>::DataArray;
  using DataArrayHost = solution_type_traits<T, 1>::DataArrayHost;
  using IndexArray = index_type_traits<1>::IndexArray;
  using IndexArrayHost = index_type_traits<1>::IndexArrayHost;
  using Matrix = jacobian_type_traits<T, 1>::JacobianMatrix;
  using MatrixHost = jacobian_type_traits<T, 1>::JacobianMatrixHost;

  inline constexpr static void
  resize(const std::array<std::size_t, 1> &n_cells,
         StructuredElementContainer<T, 1> &container) {
    Kokkos::realloc(container.node_coordinates_device, n_cells[0],
                    Basis::NNodes, 1);
    Kokkos::realloc(container.left_neighbors_device, n_cells[0], 1);
    Kokkos::realloc(container.jacobian_matrix_device, n_cells[0], Basis::NNodes,
                    1, 1);
    Kokkos::realloc(container.contravariant_vectors_device, n_cells[0],
                    Basis::NNodes, 1, 1);
    Kokkos::realloc(container.inverse_jacobian_device, n_cells[0],
                    Basis::NNodes, 1);

    Kokkos::realloc(container.node_coordinates, n_cells[0], Basis::NNodes, 1);
    Kokkos::realloc(container.left_neighbors, n_cells[0], 1);
    Kokkos::realloc(container.jacobian_matrix, n_cells[0], Basis::NNodes, 1, 1);
    Kokkos::realloc(container.contravariant_vectors, n_cells[0], Basis::NNodes,
                    1, 1);
    Kokkos::realloc(container.inverse_jacobian, n_cells[0], Basis::NNodes, 1);
  }

  inline constexpr static void
  calc_node_coordinates(const std::array<std::size_t, 1> &n_cells,
                        DataArrayHost &coordinates, const Mapping &mapping) {
    T dx = 2.0 / n_cells[0];

    for (std::size_t ielem = 0; ielem < n_cells[0]; ++ielem) {
      T cell_x_offset = -1.0 + ielem * dx + 0.5 * dx;

      for (std::size_t i = 0; i < Basis::NNodes; ++i) {
        coordinates(ielem, i, 0) =
            mapping.eval(cell_x_offset + 0.5 * dx * Basis::nodes_host[i]);
      }
    }
  }

  inline constexpr static void
  calc_jacobian_matrix(const std::array<std::size_t, 1> &n_cells,
                       MatrixHost &jacobian, const DataArrayHost &coordinates) {
    for (std::size_t ielem = 0; ielem < n_cells[0]; ++ielem) {
      for (std::size_t i = 0; i < Basis::NNodes; ++i) {
        T tmp = 0.0;
        for (std::size_t j = 0; j < Basis::NNodes; ++j) {
          tmp += Basis::derivative_host(i, j) * coordinates(ielem, j, 0);
        }
        jacobian(ielem, i, 0, 0) = tmp;
      }
    }
  }

  inline constexpr static void
  calc_inverse_jacobian(const std::array<std::size_t, 1> &n_cells,
                        DataArrayHost &inverse_jacobian,
                        const MatrixHost &jacobian) {
    for (std::size_t ielem = 0; ielem < n_cells[0]; ++ielem) {
      for (std::size_t i = 0; i < Basis::NNodes; ++i) {
        inverse_jacobian(ielem, i, 0) = 1.0 / jacobian(ielem, i, 0, 0);
      }
    }
  }

  inline constexpr static void
  initialize_left_neighbor(const std::array<std::size_t, 1> &n_cells,
                           IndexArrayHost &left_neighbors,
                           const std::array<bool, 1> &periodic) {
    for (std::size_t i = 1; i < n_cells[0]; ++i) {
      left_neighbors(i, 0) = i - 1;
    }

    if (periodic[0] == true) {
      left_neighbors(0, 0) = n_cells[0] - 1;
    } else {
      left_neighbors(0, 0) = static_cast<std::size_t>(-1);
    }
  }
};
} // namespace detail

template <class T, class Basis, class Mapping, std::size_t NDIMS>
struct StructuredElementInitializer {
  StructuredElementInitializer(Mapping mapping_,
                               const std::array<bool, NDIMS> &periodic_)
      : mapping(mapping_), periodic(periodic_) {}

  constexpr void
  init_elements(const std::array<std::size_t, NDIMS> &n_cells,
                StructuredElementContainer<T, NDIMS> &container) {
    container.nelements = n_cells;

    detail::StructuredContainerInitializer<T, Basis, Mapping, NDIMS>::resize(
        n_cells, container);

    detail::StructuredContainerInitializer<T, Basis, Mapping, NDIMS>::
        calc_node_coordinates(n_cells, container.node_coordinates, mapping);

    detail::StructuredContainerInitializer<T, Basis, Mapping, NDIMS>::
        calc_jacobian_matrix(n_cells, container.jacobian_matrix,
                             container.node_coordinates);
    std::cout << "1" << "\n";
    detail::StructuredContainerInitializer<T, Basis, Mapping, NDIMS>::
        calc_inverse_jacobian(n_cells, container.inverse_jacobian,
                              container.jacobian_matrix);

    detail::StructuredContainerInitializer<T, Basis, Mapping, NDIMS>::
        initialize_left_neighbor(n_cells, container.left_neighbors, periodic);
  }

  Mapping mapping;
  std::array<bool, NDIMS> periodic;
};
} // namespace DGSEM
