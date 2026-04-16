#pragma once

#include <Kokkos_Core.hpp>
#include <array>
#include <cstddef>
#include <utils/local_dof.hpp>

namespace DGSEM::utils {

template <std::size_t NDIMS>
struct BoundaryNodeLocation;

template <>
struct BoundaryNodeLocation<1> {
  std::size_t ielem;
  std::size_t boundary_dof;
  std::size_t storage_dof;
  std::size_t dim;
  bool left_state_inner;
};

template <>
struct BoundaryNodeLocation<2> {
  std::size_t ielem;
  std::size_t jelem;
  std::size_t boundary_dof;
  std::size_t storage_dof;
  std::size_t dim;
  bool left_state_inner;
};

template <std::size_t NDIMS, class ArrayU>
KOKKOS_INLINE_FUNCTION std::size_t infer_n_nodes(const ArrayU& u) {
  if constexpr (NDIMS == 1) {
    return u.extent(1);
  } else {
    std::size_t n_nodes = 0;
    while (n_nodes * n_nodes < u.extent(2)) {
      ++n_nodes;
    }
    return n_nodes;
  }
}

template <std::size_t NDIMS>
KOKKOS_INLINE_FUNCTION std::size_t face_dof(std::size_t n_nodes,
                                            std::size_t face,
                                            std::size_t node) {
  if constexpr (NDIMS == 1) {
    return face;
  } else {
    return face * n_nodes + node;
  }
}

template <std::size_t NDIMS, class T, class ElementData>
KOKKOS_INLINE_FUNCTION std::array<T, NDIMS>
boundary_coord(const ElementData& element_data,
               const BoundaryNodeLocation<NDIMS>& location) {
  if constexpr (NDIMS == 1) {
    return {element_data.node_coordinates(location.ielem, location.boundary_dof,
                                          0)};
  } else {
    return {
        element_data.node_coordinates(location.ielem, location.jelem,
                                      location.boundary_dof, 0),
        element_data.node_coordinates(location.ielem, location.jelem,
                                      location.boundary_dof, 1)};
  }
}

template <std::size_t NDIMS, class T, class ElementData>
KOKKOS_INLINE_FUNCTION std::array<T, NDIMS>
boundary_normal(const ElementData& element_data,
                const BoundaryNodeLocation<NDIMS>& location) {
  if constexpr (NDIMS == 1) {
    const T sign_jacobian =
        element_data.inverse_jacobian(location.ielem, location.boundary_dof) >=
                T{0}
            ? T{1}
            : T{-1};
    return {sign_jacobian *
            element_data.contravariant_vectors(location.ielem,
                                               location.boundary_dof,
                                               location.dim, 0)};
  } else {
    const T sign_jacobian =
        element_data.inverse_jacobian(location.ielem, location.jelem,
                                      location.boundary_dof) >= T{0}
            ? T{1}
            : T{-1};
    return {sign_jacobian *
                element_data.contravariant_vectors(
                    location.ielem, location.jelem, location.boundary_dof,
                    location.dim, 0),
            sign_jacobian *
                element_data.contravariant_vectors(
                    location.ielem, location.jelem, location.boundary_dof,
                    location.dim, 1)};
  }
}

template <std::size_t NDIMS, class FaceLoop>
KOKKOS_INLINE_FUNCTION void for_each_boundary_face_node(
    std::size_t face_id, int index, const std::array<std::size_t, NDIMS>& n_cells,
    std::size_t n_nodes, FaceLoop&& loop) {
  if constexpr (NDIMS == 1) {
    if (face_id == 0) {
      loop(BoundaryNodeLocation<1>{0, 0, 0, 0, false});
    } else {
      loop(BoundaryNodeLocation<1>{n_cells[0] - 1, n_nodes - 1, 1, 0, true});
    }
  } else {
    if (face_id == 0) {
      const std::size_t ielem = 0;
      const std::size_t jelem = static_cast<std::size_t>(index);
      for (std::size_t jnode = 0; jnode < n_nodes; ++jnode) {
        loop(BoundaryNodeLocation<2>{ielem, jelem, local_dof(n_nodes, 0, jnode),
                                     face_dof<2>(n_nodes, 0, jnode), 0, false});
      }
    } else if (face_id == 1) {
      const std::size_t ielem = n_cells[0] - 1;
      const std::size_t jelem = static_cast<std::size_t>(index);
      for (std::size_t jnode = 0; jnode < n_nodes; ++jnode) {
        loop(BoundaryNodeLocation<2>{
            ielem, jelem, local_dof(n_nodes, n_nodes - 1, jnode),
            face_dof<2>(n_nodes, 1, jnode), 0, true});
      }
    } else if (face_id == 2) {
      const std::size_t ielem = static_cast<std::size_t>(index);
      const std::size_t jelem = 0;
      for (std::size_t inode = 0; inode < n_nodes; ++inode) {
        loop(BoundaryNodeLocation<2>{ielem, jelem, local_dof(n_nodes, inode, 0),
                                     face_dof<2>(n_nodes, 2, inode), 1, false});
      }
    } else {
      const std::size_t ielem = static_cast<std::size_t>(index);
      const std::size_t jelem = n_cells[1] - 1;
      for (std::size_t inode = 0; inode < n_nodes; ++inode) {
        loop(BoundaryNodeLocation<2>{
            ielem, jelem, local_dof(n_nodes, inode, n_nodes - 1),
            face_dof<2>(n_nodes, 3, inode), 1, true});
      }
    }
  }
}

} // namespace DGSEM::utils
