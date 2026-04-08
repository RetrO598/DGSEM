#pragma once

#include <Kokkos_Core.hpp>
#include <Kokkos_Macros.hpp>
#include <array>
#include <base/base.hpp>
#include <cstddef>
#include <equations/equations.hpp>
#include <utils/local_dof.hpp>

namespace DGSEM {
template <class T, std::size_t NDIMS>
struct StructuredElementContainer {
  using CoordArray = coordinate_type_traits<T, NDIMS>::CoordArray;
  using CoordArrayHost = coordinate_type_traits<T, NDIMS>::CoordArrayHost;
  using ScalarArray = scalar_node_type_traits<T, NDIMS>::ScalarArray;
  using ScalarArrayHost = scalar_node_type_traits<T, NDIMS>::ScalarArrayHost;
  using IndexArray = index_type_traits<NDIMS>::IndexArray;
  using IndexArrayHost = index_type_traits<NDIMS>::IndexArrayHost;
  using Matrix = jacobian_type_traits<T, NDIMS>::JacobianMatrix;
  using MatrixHost = jacobian_type_traits<T, NDIMS>::JacobianMatrixHost;

  std::array<std::size_t, NDIMS> nelements;
  CoordArrayHost node_coordinates;
  IndexArrayHost left_neighbors;
  MatrixHost jacobian_matrix;
  MatrixHost contravariant_vectors;
  ScalarArrayHost inverse_jacobian;
  SubcellNormalVectors<T, NDIMS> subcell_normals;

  CoordArray node_coordinates_device;
  IndexArray left_neighbors_device;
  Matrix jacobian_matrix_device;
  Matrix contravariant_vectors_device;
  ScalarArray inverse_jacobian_device;

  void sync_to_device() {
    Kokkos::deep_copy(node_coordinates_device, node_coordinates);
    Kokkos::deep_copy(left_neighbors_device, left_neighbors);
    Kokkos::deep_copy(jacobian_matrix_device, jacobian_matrix);
    Kokkos::deep_copy(contravariant_vectors_device, contravariant_vectors);
    Kokkos::deep_copy(inverse_jacobian_device, inverse_jacobian);
    subcell_normals.sync_to_device();
  }
};

namespace detail {
template <class T, class Basis, class Mapping, std::size_t NDIMS>
struct StructuredContainerInitializer;

template <class T, class Basis, class Mapping>
struct StructuredContainerInitializer<T, Basis, Mapping, 1> {
  using CoordArrayHost = coordinate_type_traits<T, 1>::CoordArrayHost;
  using ScalarArrayHost = scalar_node_type_traits<T, 1>::ScalarArrayHost;
  using IndexArray = index_type_traits<1>::IndexArray;
  using IndexArrayHost = index_type_traits<1>::IndexArrayHost;
  using Matrix = jacobian_type_traits<T, 1>::JacobianMatrix;
  using MatrixHost = jacobian_type_traits<T, 1>::JacobianMatrixHost;

  inline constexpr static void
  resize(const std::array<std::size_t, 1>& n_cells,
         StructuredElementContainer<T, 1>& container) {
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
  calc_node_coordinates(const std::array<std::size_t, 1>& n_cells,
                        CoordArrayHost& coordinates, const Mapping& mapping) {
    const T dx = static_cast<T>(2.0) / static_cast<T>(n_cells[0]);

    for (std::size_t ielem = 0; ielem < n_cells[0]; ++ielem) {
      const T cell_x_offset = static_cast<T>(-1.0) +
                              static_cast<T>(ielem) * dx +
                              static_cast<T>(0.5) * dx;

      for (std::size_t i = 0; i < Basis::NNodes; ++i) {
        coordinates(ielem, i, 0) = mapping.eval(
            cell_x_offset + static_cast<T>(0.5) * dx * Basis::nodes_host[i]);
      }
    }
  }

  inline constexpr static void
  calc_jacobian_matrix(const std::array<std::size_t, 1>& n_cells,
                       MatrixHost& jacobian,
                       const CoordArrayHost& coordinates) {
    for (std::size_t ielem = 0; ielem < n_cells[0]; ++ielem) {
      for (std::size_t i = 0; i < Basis::NNodes; ++i) {
        T tmp = T{0.0};
        for (std::size_t j = 0; j < Basis::NNodes; ++j) {
          tmp += Basis::derivative_host(i, j) * coordinates(ielem, j, 0);
        }
        jacobian(ielem, i, 0, 0) = tmp;
      }
    }
  }

  inline constexpr static void
  calc_contravariant_vectors(const std::array<std::size_t, 1>& n_cells,
                             MatrixHost& contravariant,
                             const MatrixHost& jacobian) {}

  inline constexpr static void
  calc_inverse_jacobian(const std::array<std::size_t, 1>& n_cells,
                        ScalarArrayHost& inverse_jacobian,
                        const MatrixHost& jacobian) {
    for (std::size_t ielem = 0; ielem < n_cells[0]; ++ielem) {
      for (std::size_t i = 0; i < Basis::NNodes; ++i) {
        inverse_jacobian(ielem, i) =
            static_cast<T>(1.0) / jacobian(ielem, i, 0, 0);
      }
    }
  }

  inline constexpr static void
  initialize_left_neighbor(const std::array<std::size_t, 1>& n_cells,
                           IndexArrayHost& left_neighbors,
                           const std::array<bool, 1>& periodic) {
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

template <class T, class Basis, class Mapping>
struct StructuredContainerInitializer<T, Basis, Mapping, 2> {
  using CoordArrayHost = coordinate_type_traits<T, 2>::CoordArrayHost;
  using ScalarArrayHost = scalar_node_type_traits<T, 2>::ScalarArrayHost;
  using IndexArray = index_type_traits<2>::IndexArray;
  using IndexArrayHost = index_type_traits<2>::IndexArrayHost;
  using Matrix = jacobian_type_traits<T, 2>::JacobianMatrix;
  using MatrixHost = jacobian_type_traits<T, 2>::JacobianMatrixHost;
  inline constexpr static std::size_t ndofs() {
    return Basis::NNodes * Basis::NNodes;
  }

  inline constexpr static void
  resize(const std::array<std::size_t, 2>& n_cells,
         StructuredElementContainer<T, 2>& container) {
    Kokkos::realloc(container.node_coordinates_device, n_cells[0], n_cells[1],
                    ndofs(), 2);
    Kokkos::realloc(container.left_neighbors_device, n_cells[0], n_cells[1], 2,
                    2);
    Kokkos::realloc(container.jacobian_matrix_device, n_cells[0], n_cells[1],
                    ndofs(), 2, 2);
    Kokkos::realloc(container.contravariant_vectors_device, n_cells[0],
                    n_cells[1], ndofs(), 2, 2);
    Kokkos::realloc(container.inverse_jacobian_device, n_cells[0], n_cells[1],
                    ndofs());

    Kokkos::realloc(container.node_coordinates, n_cells[0], n_cells[1], ndofs(),
                    2);
    Kokkos::realloc(container.left_neighbors, n_cells[0], n_cells[1], 2, 2);
    Kokkos::realloc(container.jacobian_matrix, n_cells[0], n_cells[1], ndofs(),
                    2, 2);
    Kokkos::realloc(container.contravariant_vectors, n_cells[0], n_cells[1],
                    ndofs(), 2, 2);
    Kokkos::realloc(container.inverse_jacobian, n_cells[0], n_cells[1],
                    ndofs());
    container.subcell_normals.allocate(n_cells, Basis::NNodes);
  }

  inline constexpr static void
  calc_node_coordinates(const std::array<std::size_t, 2>& n_cells,
                        CoordArrayHost& coordinates, const Mapping& mapping) {
    const T dx = static_cast<T>(2.0) / static_cast<T>(n_cells[0]);
    const T dy = static_cast<T>(2.0) / static_cast<T>(n_cells[1]);

    for (std::size_t ielem = 0; ielem < n_cells[0]; ++ielem) {
      const T cell_x_offset = static_cast<T>(-1.0) +
                              static_cast<T>(ielem) * dx +
                              static_cast<T>(0.5) * dx;
      for (std::size_t jelem = 0; jelem < n_cells[1]; ++jelem) {
        const T cell_y_offset = static_cast<T>(-1.0) +
                                static_cast<T>(jelem) * dy +
                                static_cast<T>(0.5) * dy;
        for (std::size_t jnode = 0; jnode < Basis::NNodes; ++jnode) {
          for (std::size_t inode = 0; inode < Basis::NNodes; ++inode) {
            const std::size_t dof =
                DGSEM::utils::local_dof<Basis::NNodes>(inode, jnode);
            const std::array<T, 2> ref_coord{
                cell_x_offset +
                    static_cast<T>(0.5) * dx * Basis::nodes_host[inode],
                cell_y_offset +
                    static_cast<T>(0.5) * dy * Basis::nodes_host[jnode]};
            const auto phys_coord = mapping.eval(ref_coord);
            coordinates(ielem, jelem, dof, 0) = phys_coord[0];
            coordinates(ielem, jelem, dof, 1) = phys_coord[1];
          }
        }
      }
    }
  }

  inline constexpr static void
  calc_jacobian_matrix(const std::array<std::size_t, 2>& n_cells,
                       MatrixHost& jacobian,
                       const CoordArrayHost& coordinates) {
    for (std::size_t ielem = 0; ielem < n_cells[0]; ++ielem) {
      for (std::size_t jelem = 0; jelem < n_cells[1]; ++jelem) {
        for (std::size_t jnode = 0; jnode < Basis::NNodes; ++jnode) {
          for (std::size_t inode = 0; inode < Basis::NNodes; ++inode) {
            const std::size_t dof =
                DGSEM::utils::local_dof<Basis::NNodes>(inode, jnode);
            T dx_dxi = T{};
            T dy_dxi = T{};
            T dx_deta = T{};
            T dy_deta = T{};

            for (std::size_t k = 0; k < Basis::NNodes; ++k) {
              const std::size_t dof_x =
                  DGSEM::utils::local_dof<Basis::NNodes>(k, jnode);
              const std::size_t dof_y =
                  DGSEM::utils::local_dof<Basis::NNodes>(inode, k);
              dx_dxi += Basis::derivative_host(inode, k) *
                        coordinates(ielem, jelem, dof_x, 0);
              dy_dxi += Basis::derivative_host(inode, k) *
                        coordinates(ielem, jelem, dof_x, 1);
              dx_deta += Basis::derivative_host(jnode, k) *
                         coordinates(ielem, jelem, dof_y, 0);
              dy_deta += Basis::derivative_host(jnode, k) *
                         coordinates(ielem, jelem, dof_y, 1);
            }

            jacobian(ielem, jelem, dof, 0, 0) = dx_dxi;
            jacobian(ielem, jelem, dof, 0, 1) = dy_dxi;
            jacobian(ielem, jelem, dof, 1, 0) = dx_deta;
            jacobian(ielem, jelem, dof, 1, 1) = dy_deta;
          }
        }
      }
    }
  }

  inline constexpr static void
  calc_contravariant_vectors(const std::array<std::size_t, 2>& n_cells,
                             MatrixHost& contravariant,
                             const MatrixHost& jacobian) {
    for (std::size_t ielem = 0; ielem < n_cells[0]; ++ielem) {
      for (std::size_t jelem = 0; jelem < n_cells[1]; ++jelem) {
        for (std::size_t jnode = 0; jnode < Basis::NNodes; ++jnode) {
          for (std::size_t inode = 0; inode < Basis::NNodes; ++inode) {
            const std::size_t dof =
                DGSEM::utils::local_dof<Basis::NNodes>(inode, jnode);
            contravariant(ielem, jelem, dof, 0, 0) =
                jacobian(ielem, jelem, dof, 1, 1);
            contravariant(ielem, jelem, dof, 0, 1) =
                -jacobian(ielem, jelem, dof, 1, 0);
            contravariant(ielem, jelem, dof, 1, 0) =
                -jacobian(ielem, jelem, dof, 0, 1);
            contravariant(ielem, jelem, dof, 1, 1) =
                jacobian(ielem, jelem, dof, 0, 0);
          }
        }
      }
    }
  }

  inline constexpr static void
  calc_inverse_jacobian(const std::array<std::size_t, 2>& n_cells,
                        ScalarArrayHost& inverse_jacobian,
                        const MatrixHost& jacobian) {
    for (std::size_t ielem = 0; ielem < n_cells[0]; ++ielem) {
      for (std::size_t jelem = 0; jelem < n_cells[1]; ++jelem) {
        for (std::size_t jnode = 0; jnode < Basis::NNodes; ++jnode) {
          for (std::size_t inode = 0; inode < Basis::NNodes; ++inode) {
            const std::size_t dof =
                DGSEM::utils::local_dof<Basis::NNodes>(inode, jnode);
            inverse_jacobian(ielem, jelem, dof) =
                1.0 / (jacobian(ielem, jelem, dof, 0, 0) *
                           jacobian(ielem, jelem, dof, 1, 1) -
                       jacobian(ielem, jelem, dof, 1, 0) *
                           jacobian(ielem, jelem, dof, 0, 1));
          }
        }
      }
    }
  }

  inline constexpr static void
  initialize_left_neighbor(const std::array<std::size_t, 2>& n_cells,
                           IndexArrayHost& neighbors,
                           const std::array<bool, 2>& periodic) {
    constexpr std::size_t invalid = static_cast<std::size_t>(-1);

    for (std::size_t ielem = 0; ielem < n_cells[0]; ++ielem) {
      for (std::size_t jelem = 0; jelem < n_cells[1]; ++jelem) {
        neighbors(ielem, jelem, 0, 0) =
            (ielem > 0) ? ielem - 1 : (periodic[0] ? n_cells[0] - 1 : invalid);
        neighbors(ielem, jelem, 0, 1) = jelem;

        neighbors(ielem, jelem, 1, 0) = ielem;
        neighbors(ielem, jelem, 1, 1) =
            (jelem > 0) ? jelem - 1 : (periodic[1] ? n_cells[1] - 1 : invalid);
      }
    }
  }
};
} // namespace detail

template <class T, class Basis, class Mapping, std::size_t NDIMS>
struct StructuredElementInitializer {
  StructuredElementInitializer(Mapping mapping_,
                               const std::array<bool, NDIMS>& periodic_)
      : mapping(mapping_), periodic(periodic_) {}

  constexpr void
  init_elements(const std::array<std::size_t, NDIMS>& n_cells,
                StructuredElementContainer<T, NDIMS>& container) {
    container.nelements = n_cells;

    detail::StructuredContainerInitializer<T, Basis, Mapping, NDIMS>::resize(
        n_cells, container);

    detail::StructuredContainerInitializer<T, Basis, Mapping, NDIMS>::
        calc_node_coordinates(n_cells, container.node_coordinates, mapping);

    detail::StructuredContainerInitializer<T, Basis, Mapping, NDIMS>::
        calc_jacobian_matrix(n_cells, container.jacobian_matrix,
                             container.node_coordinates);

    detail::StructuredContainerInitializer<T, Basis, Mapping, NDIMS>::
        calc_contravariant_vectors(n_cells, container.contravariant_vectors,
                                   container.jacobian_matrix);

    detail::StructuredContainerInitializer<T, Basis, Mapping, NDIMS>::
        calc_inverse_jacobian(n_cells, container.inverse_jacobian,
                              container.jacobian_matrix);

    if constexpr (NDIMS == 2) {
      container.subcell_normals.template initialize_host<Basis>(
          n_cells, container.contravariant_vectors);
    }

    detail::StructuredContainerInitializer<T, Basis, Mapping, NDIMS>::
        initialize_left_neighbor(n_cells, container.left_neighbors, periodic);

    container.sync_to_device();
  }

  Mapping mapping;
  std::array<bool, NDIMS> periodic;
};
} // namespace DGSEM
