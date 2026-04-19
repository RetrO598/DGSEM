#pragma once
#include <Kokkos_Core.hpp>
#include <Kokkos_Core_fwd.hpp>
#include <array>
#include <cstddef>
#include <utils/local_dof.hpp>

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

template <class T>
struct solution_type_traits<T, 3> {
  using DataArray = Kokkos::View<T*****, Device>;
  using DataArrayHost = DataArray::HostMirror;
  using ElementArray = Kokkos::View<T***, Device>;
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

template <class T>
struct coordinate_type_traits<T, 3> {
  using CoordArray = Kokkos::View<T*****, Device>;
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

template <class T>
struct scalar_node_type_traits<T, 3> {
  using ScalarArray = Kokkos::View<T****, Device>;
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

template <>
struct index_type_traits<3> {
  using IndexArray = Kokkos::View<std::size_t*****, Device>;
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

template <class T>
struct jacobian_type_traits<T, 3> {
  using JacobianMatrix = Kokkos::View<T******, Device>;
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
    for (std::size_t ielem = 0; ielem < n_cells[0]; ++ielem) {
      for (std::size_t jelem = 0; jelem < n_cells[1]; ++jelem) {
        for (std::size_t jnode = 0; jnode < Basis::NNodes; ++jnode) {
          const std::size_t first_dof =
              DGSEM::utils::local_dof<Basis::NNodes>(0, jnode);
          normal_1_host(ielem, jelem, 0, jnode, 0) =
              contravariant_vectors(ielem, jelem, first_dof, 0, 0);
          normal_1_host(ielem, jelem, 0, jnode, 1) =
              contravariant_vectors(ielem, jelem, first_dof, 0, 1);

          for (std::size_t m = 0; m < Basis::NNodes; ++m) {
            const std::size_t dof_m =
                DGSEM::utils::local_dof<Basis::NNodes>(m, jnode);
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
              const std::size_t dof_m =
                  DGSEM::utils::local_dof<Basis::NNodes>(m, jnode);
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
          const std::size_t first_dof =
              DGSEM::utils::local_dof<Basis::NNodes>(inode, 0);
          normal_2_host(ielem, jelem, inode, 0, 0) =
              contravariant_vectors(ielem, jelem, first_dof, 1, 0);
          normal_2_host(ielem, jelem, inode, 0, 1) =
              contravariant_vectors(ielem, jelem, first_dof, 1, 1);

          for (std::size_t m = 0; m < Basis::NNodes; ++m) {
            const std::size_t dof_m =
                DGSEM::utils::local_dof<Basis::NNodes>(inode, m);
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
              const std::size_t dof_m =
                  DGSEM::utils::local_dof<Basis::NNodes>(inode, m);
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

  template <class Basis, class MatrixHost>
  void initialize_host(const std::array<std::size_t, 3>& n_cells,
                       const MatrixHost& contravariant_vectors) {
    for (std::size_t ielem = 0; ielem < n_cells[0]; ++ielem) {
      for (std::size_t jelem = 0; jelem < n_cells[1]; ++jelem) {
        for (std::size_t kelem = 0; kelem < n_cells[2]; ++kelem) {
          for (std::size_t jnode = 0; jnode < Basis::NNodes; ++jnode) {
            for (std::size_t knode = 0; knode < Basis::NNodes; ++knode) {
              const std::size_t first_dof =
                  DGSEM::utils::local_dof<Basis::NNodes>(0, jnode, knode);

              for (std::size_t dim = 0; dim < 3; ++dim) {
                normal_1_host(ielem, jelem, kelem, 0, jnode, knode, dim) =
                    contravariant_vectors(ielem, jelem, kelem, first_dof, 0,
                                          dim);
              }

              for (std::size_t m = 0; m < Basis::NNodes; ++m) {
                const std::size_t dof_m =
                    DGSEM::utils::local_dof<Basis::NNodes>(m, jnode, knode);
                const T wD =
                    Basis::weights_host[0] * Basis::derivative_host(0, m);
                for (std::size_t dim = 0; dim < 3; ++dim) {
                  normal_1_host(ielem, jelem, kelem, 0, jnode, knode, dim) +=
                      wD *
                      contravariant_vectors(ielem, jelem, kelem, dof_m, 0, dim);
                }
              }

              for (std::size_t iface = 1; iface < Basis::NNodes - 1; ++iface) {
                for (std::size_t dim = 0; dim < 3; ++dim) {
                  normal_1_host(ielem, jelem, kelem, iface, jnode, knode, dim) =
                      normal_1_host(ielem, jelem, kelem, iface - 1, jnode,
                                    knode, dim);
                }
                for (std::size_t m = 0; m < Basis::NNodes; ++m) {
                  const std::size_t dof_m =
                      DGSEM::utils::local_dof<Basis::NNodes>(m, jnode, knode);
                  const T wD = Basis::weights_host[iface] *
                               Basis::derivative_host(iface, m);
                  for (std::size_t dim = 0; dim < 3; ++dim) {
                    normal_1_host(ielem, jelem, kelem, iface, jnode, knode,
                                  dim) +=
                        wD * contravariant_vectors(ielem, jelem, kelem, dof_m,
                                                   0, dim);
                  }
                }
              }
            }
          }

          for (std::size_t inode = 0; inode < Basis::NNodes; ++inode) {
            for (std::size_t knode = 0; knode < Basis::NNodes; ++knode) {
              const std::size_t first_dof =
                  DGSEM::utils::local_dof<Basis::NNodes>(inode, 0, knode);
              for (std::size_t dim = 0; dim < 3; ++dim) {
                normal_2_host(ielem, jelem, kelem, inode, 0, knode, dim) =
                    contravariant_vectors(ielem, jelem, kelem, first_dof, 1,
                                          dim);
              }

              for (std::size_t m = 0; m < Basis::NNodes; ++m) {
                const std::size_t dof_m =
                    DGSEM::utils::local_dof<Basis::NNodes>(inode, m, knode);
                const T wD =
                    Basis::weights_host[0] * Basis::derivative_host(0, m);
                for (std::size_t dim = 0; dim < 3; ++dim) {
                  normal_2_host(ielem, jelem, kelem, inode, 0, knode, dim) +=
                      wD *
                      contravariant_vectors(ielem, jelem, kelem, dof_m, 1, dim);
                }
              }

              for (std::size_t jface = 1; jface < Basis::NNodes - 1; ++jface) {
                for (std::size_t dim = 0; dim < 3; ++dim) {
                  normal_2_host(ielem, jelem, kelem, inode, jface, knode, dim) =
                      normal_2_host(ielem, jelem, kelem, inode, jface - 1,
                                    knode, dim);
                }
                for (std::size_t m = 0; m < Basis::NNodes; ++m) {
                  const std::size_t dof_m =
                      DGSEM::utils::local_dof<Basis::NNodes>(inode, m, knode);
                  const T wD = Basis::weights_host[jface] *
                               Basis::derivative_host(jface, m);
                  for (std::size_t dim = 0; dim < 3; ++dim) {
                    normal_2_host(ielem, jelem, kelem, inode, jface, knode,
                                  dim) +=
                        wD * contravariant_vectors(ielem, jelem, kelem, dof_m,
                                                   1, dim);
                  }
                }
              }
            }
          }

          for (std::size_t inode = 0; inode < Basis::NNodes; ++inode) {
            for (std::size_t jnode = 0; jnode < Basis::NNodes; ++jnode) {
              const std::size_t first_dof =
                  DGSEM::utils::local_dof<Basis::NNodes>(inode, jnode, 0);
              for (std::size_t dim = 0; dim < 3; ++dim) {
                normal_3_host(ielem, jelem, kelem, inode, jnode, 0, dim) =
                    contravariant_vectors(ielem, jelem, kelem, first_dof, 2,
                                          dim);
              }

              for (std::size_t m = 0; m < Basis::NNodes; ++m) {
                const std::size_t dof_m =
                    DGSEM::utils::local_dof<Basis::NNodes>(inode, jnode, m);
                const T wD =
                    Basis::weights_host[0] * Basis::derivative_host(0, m);
                for (std::size_t dim = 0; dim < 3; ++dim) {
                  normal_3_host(ielem, jelem, kelem, inode, jnode, 0, dim) +=
                      wD *
                      contravariant_vectors(ielem, jelem, kelem, dof_m, 2, dim);
                }
              }

              for (std::size_t kface = 1; kface < Basis::NNodes - 1; ++kface) {
                for (std::size_t dim = 0; dim < 3; ++dim) {
                  normal_3_host(ielem, jelem, kelem, inode, jnode, kface, dim) =
                      normal_3_host(ielem, jelem, kelem, inode, jnode,
                                    kface - 1, dim);
                }
                for (std::size_t m = 0; m < Basis::NNodes; ++m) {
                  const std::size_t dof_m =
                      DGSEM::utils::local_dof<Basis::NNodes>(inode, jnode, m);
                  const T wD = Basis::weights_host[kface] *
                               Basis::derivative_host(kface, m);
                  for (std::size_t dim = 0; dim < 3; ++dim) {
                    normal_3_host(ielem, jelem, kelem, inode, jnode, kface,
                                  dim) +=
                        wD * contravariant_vectors(ielem, jelem, kelem, dof_m,
                                                   2, dim);
                  }
                }
              }
            }
          }
        }
      }
    }
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
