#pragma once
#include <Kokkos_Core.hpp>
#include <Kokkos_Core_fwd.hpp>
#include <array>
#include <cstddef>

namespace DGSEM {
using Device = Kokkos::Device<Kokkos::DefaultExecutionSpace,
                              Kokkos::DefaultExecutionSpace::memory_space>;

template <class T, std::size_t NDIMS>
struct solution_type_traits {};

template <class T>
struct solution_type_traits<T, 1> {
  using DataArray = Kokkos::View<T***, Device>;
  using DataArrayHost = DataArray::HostMirror;
  using ElementArray = Kokkos::View<T*, Device>;
  using ElementArrayHost = ElementArray::HostMirror;
};

template <class T>
struct solution_type_traits<T, 2> {
  using DataArray = Kokkos::View<T****, Device>;
  using DataArrayHost = DataArray::HostMirror;
  using ElementArray = Kokkos::View<T**, Device>;
  using ElementArrayHost = ElementArray::HostMirror;
};

template <class T, std::size_t NDIMS>
struct coordinate_type_traits {};

template <class T>
struct coordinate_type_traits<T, 1> {
  using CoordArray = Kokkos::View<T***, Device>;
  using CoordArrayHost = CoordArray::HostMirror;
};

template <class T>
struct coordinate_type_traits<T, 2> {
  using CoordArray = Kokkos::View<T****, Device>;
  using CoordArrayHost = CoordArray::HostMirror;
};

template <class T, std::size_t NDIMS>
struct scalar_node_type_traits {};

template <class T>
struct scalar_node_type_traits<T, 1> {
  using ScalarArray = Kokkos::View<T**, Device>;
  using ScalarArrayHost = ScalarArray::HostMirror;
};

template <class T>
struct scalar_node_type_traits<T, 2> {
  using ScalarArray = Kokkos::View<T***, Device>;
  using ScalarArrayHost = ScalarArray::HostMirror;
};

template <std::size_t NDIMS>
struct index_type_traits {};

template <>
struct index_type_traits<1> {
  using IndexArray = Kokkos::View<std::size_t**, Device>;
  using IndexArrayHost = IndexArray::HostMirror;
};

template <>
struct index_type_traits<2> {
  using IndexArray = Kokkos::View<std::size_t****, Device>;
  using IndexArrayHost = IndexArray::HostMirror;
};

template <class T, std::size_t NDIMS>
struct jacobian_type_traits {};

template <class T>
struct jacobian_type_traits<T, 1> {
  using JacobianMatrix = Kokkos::View<T****, Device>;
  using JacobianMatrixHost = JacobianMatrix::HostMirror;
};

template <class T>
struct jacobian_type_traits<T, 2> {
  using JacobianMatrix = Kokkos::View<T*****, Device>;
  using JacobianMatrixHost = JacobianMatrix::HostMirror;
};

template <class T, std::size_t NDIMS>
struct SubcellNormalVectors {
  struct DeviceData {};

  void sync_to_device() {}
  DeviceData device_data() const { return {}; }
};

template <class T>
struct SubcellNormalVectors<T, 2> {
  using NormalArray1 = Kokkos::View<T*****, Device>;
  using NormalArray2 = Kokkos::View<T*****, Device>;
  using NormalArray1Host = typename NormalArray1::HostMirror;
  using NormalArray2Host = typename NormalArray2::HostMirror;

  struct DeviceData {
    NormalArray1 normal_1;
    NormalArray2 normal_2;
  };

  void allocate(const std::array<std::size_t, 2>& n_cells,
                std::size_t n_nodes) {
    Kokkos::realloc(normal_1, n_cells[0], n_cells[1], n_nodes - 1, n_nodes, 2);
    Kokkos::realloc(normal_2, n_cells[0], n_cells[1], n_nodes, n_nodes - 1, 2);
    Kokkos::realloc(normal_1_host, n_cells[0], n_cells[1], n_nodes - 1, n_nodes,
                    2);
    Kokkos::realloc(normal_2_host, n_cells[0], n_cells[1], n_nodes, n_nodes - 1,
                    2);
  }

  template <class Basis, class MatrixHost>
  void initialize_host(const std::array<std::size_t, 2>& n_cells,
                       const MatrixHost& contravariant_vectors) {
    auto local_dof = [](std::size_t inode, std::size_t jnode) {
      return jnode * Basis::NNodes + inode;
    };

    for (std::size_t ielem = 0; ielem < n_cells[0]; ++ielem) {
      for (std::size_t jelem = 0; jelem < n_cells[1]; ++jelem) {
        for (std::size_t jnode = 0; jnode < Basis::NNodes; ++jnode) {
          const std::size_t first_dof = local_dof(0, jnode);
          normal_1_host(ielem, jelem, 0, jnode, 0) =
              contravariant_vectors(ielem, jelem, first_dof, 0, 0);
          normal_1_host(ielem, jelem, 0, jnode, 1) =
              contravariant_vectors(ielem, jelem, first_dof, 0, 1);

          for (std::size_t m = 0; m < Basis::NNodes; ++m) {
            const std::size_t dof_m = local_dof(m, jnode);
            const T wD = Basis::weights_host[0] * Basis::derivative_host(0, m);
            normal_1_host(ielem, jelem, 0, jnode, 0) +=
                wD * contravariant_vectors(ielem, jelem, dof_m, 0, 0);
            normal_1_host(ielem, jelem, 0, jnode, 1) +=
                wD * contravariant_vectors(ielem, jelem, dof_m, 0, 1);
          }

          for (std::size_t iface = 1; iface < Basis::NNodes - 1; ++iface) {
            normal_1_host(ielem, jelem, iface, jnode, 0) =
                normal_1_host(ielem, jelem, iface - 1, jnode, 0);
            normal_1_host(ielem, jelem, iface, jnode, 1) =
                normal_1_host(ielem, jelem, iface - 1, jnode, 1);
            for (std::size_t m = 0; m < Basis::NNodes; ++m) {
              const std::size_t dof_m = local_dof(m, jnode);
              const T wD =
                  Basis::weights_host[iface] * Basis::derivative_host(iface, m);
              normal_1_host(ielem, jelem, iface, jnode, 0) +=
                  wD * contravariant_vectors(ielem, jelem, dof_m, 0, 0);
              normal_1_host(ielem, jelem, iface, jnode, 1) +=
                  wD * contravariant_vectors(ielem, jelem, dof_m, 0, 1);
            }
          }
        }

        for (std::size_t inode = 0; inode < Basis::NNodes; ++inode) {
          const std::size_t first_dof = local_dof(inode, 0);
          normal_2_host(ielem, jelem, inode, 0, 0) =
              contravariant_vectors(ielem, jelem, first_dof, 1, 0);
          normal_2_host(ielem, jelem, inode, 0, 1) =
              contravariant_vectors(ielem, jelem, first_dof, 1, 1);

          for (std::size_t m = 0; m < Basis::NNodes; ++m) {
            const std::size_t dof_m = local_dof(inode, m);
            const T wD = Basis::weights_host[0] * Basis::derivative_host(0, m);
            normal_2_host(ielem, jelem, inode, 0, 0) +=
                wD * contravariant_vectors(ielem, jelem, dof_m, 1, 0);
            normal_2_host(ielem, jelem, inode, 0, 1) +=
                wD * contravariant_vectors(ielem, jelem, dof_m, 1, 1);
          }

          for (std::size_t iface = 1; iface < Basis::NNodes - 1; ++iface) {
            normal_2_host(ielem, jelem, inode, iface, 0) =
                normal_2_host(ielem, jelem, inode, iface - 1, 0);
            normal_2_host(ielem, jelem, inode, iface, 1) =
                normal_2_host(ielem, jelem, inode, iface - 1, 1);
            for (std::size_t m = 0; m < Basis::NNodes; ++m) {
              const std::size_t dof_m = local_dof(inode, m);
              const T wD =
                  Basis::weights_host[iface] * Basis::derivative_host(iface, m);
              normal_2_host(ielem, jelem, inode, iface, 0) +=
                  wD * contravariant_vectors(ielem, jelem, dof_m, 1, 0);
              normal_2_host(ielem, jelem, inode, iface, 1) +=
                  wD * contravariant_vectors(ielem, jelem, dof_m, 1, 1);
            }
          }
        }
      }
    }
  }

  void sync_to_device() {
    Kokkos::deep_copy(normal_1, normal_1_host);
    Kokkos::deep_copy(normal_2, normal_2_host);
  }

  DeviceData device_data() const { return DeviceData{normal_1, normal_2}; }

  NormalArray1Host normal_1_host;
  NormalArray2Host normal_2_host;
  NormalArray1 normal_1;
  NormalArray2 normal_2;
};

template <class T>
struct SubcellNormalVectors<T, 3> {
  using NormalArray1 = Kokkos::View<T*******, Device>;
  using NormalArray2 = Kokkos::View<T*******, Device>;
  using NormalArray3 = Kokkos::View<T*******, Device>;
  using NormalArray1Host = typename NormalArray1::HostMirror;
  using NormalArray2Host = typename NormalArray2::HostMirror;
  using NormalArray3Host = typename NormalArray3::HostMirror;

  struct DeviceData {
    NormalArray1 normal_1;
    NormalArray2 normal_2;
    NormalArray3 normal_3;
  };

  void allocate(const std::array<std::size_t, 3>& n_cells,
                std::size_t n_nodes) {
    Kokkos::realloc(normal_1, n_cells[0], n_cells[1], n_cells[2], n_nodes - 1,
                    n_nodes, n_nodes, 3);
    Kokkos::realloc(normal_2, n_cells[0], n_cells[1], n_cells[2], n_nodes,
                    n_nodes - 1, n_nodes, 3);
    Kokkos::realloc(normal_3, n_cells[0], n_cells[1], n_cells[2], n_nodes,
                    n_nodes, n_nodes - 1, 3);
    Kokkos::realloc(normal_1_host, n_cells[0], n_cells[1], n_cells[2],
                    n_nodes - 1, n_nodes, n_nodes, 3);
    Kokkos::realloc(normal_2_host, n_cells[0], n_cells[1], n_cells[2], n_nodes,
                    n_nodes - 1, n_nodes, 3);
    Kokkos::realloc(normal_3_host, n_cells[0], n_cells[1], n_cells[2], n_nodes,
                    n_nodes, n_nodes - 1, 3);
  }

  void sync_to_device() {
    Kokkos::deep_copy(normal_1, normal_1_host);
    Kokkos::deep_copy(normal_2, normal_2_host);
    Kokkos::deep_copy(normal_3, normal_3_host);
  }

  DeviceData device_data() const {
    return DeviceData{normal_1, normal_2, normal_3};
  }

  NormalArray1Host normal_1_host;
  NormalArray2Host normal_2_host;
  NormalArray3Host normal_3_host;
  NormalArray1 normal_1;
  NormalArray2 normal_2;
  NormalArray3 normal_3;
};
} // namespace DGSEM
