#pragma once

#include <cstddef>
#include <polynomial_basis/basis_detail.hpp>
#include <span>

namespace DGSEM {
namespace Basis {

template <class T, std::size_t Polydeg>
class LobattoLegendreBasis {
 public:
  static constexpr std::size_t NNodes = Polydeg + 1;

  using View1D = Kokkos::View<T*, Device>;
  using View2D = Kokkos::View<T**, Device>;

  struct DeviceData {
    static constexpr std::size_t NNodes = Polydeg + 1;

    View1D nodes;
    View1D weights;
    View1D inv_weights;
    View1D boundary_interpolation_left;
    View1D boundary_interpolation_right;
    View2D derivative_matrix;
    View2D derivative_split;
    View2D derivative_split_transpose;
    View2D derivative_dhat;
    View2D inverse_vandermonde_legendre;
  };

  inline static View1D nodes;
  inline static View1D weights;
  inline static View1D inv_weights;
  inline static View1D boundary_interpolation_left;
  inline static View1D boundary_interpolation_right;
  inline static View2D derivative_matrix;
  inline static View2D derivative_split;
  inline static View2D derivative_split_transpose;
  inline static View2D derivative_dhat;
  inline static View2D inverse_vandermonde_legendre;

  static void initialize() {
    if (nodes.is_allocated()) return;

    auto data_host = detail::GLL::gauss_lobatto_nodes_weights<T, NNodes>();

    nodes = View1D("nodes", NNodes);
    weights = View1D("weights", NNodes);
    inv_weights = View1D("inv_weights", NNodes);
    boundary_interpolation_left = View1D("boundary_interpolation_left", NNodes);
    boundary_interpolation_right =
        View1D("boundary_interpolation_right", NNodes);
    derivative_matrix = View2D("derivative_matrix", NNodes, NNodes);
    derivative_split = View2D("derivative_split", NNodes, NNodes);
    derivative_split_transpose =
        View2D("derivative_split_transpose", NNodes, NNodes);
    derivative_dhat = View2D("derivative_dhat", NNodes, NNodes);
    inverse_vandermonde_legendre =
        View2D("inverse_vandermonde_legendre", NNodes, NNodes);

    auto nodes_h = Kokkos::create_mirror_view(nodes);
    auto weights_h = Kokkos::create_mirror_view(weights);
    auto inv_weights_h = Kokkos::create_mirror_view(inv_weights);

    for (std::size_t i = 0; i < NNodes; ++i) {
      nodes_h(i) = data_host.nodes[i];
      weights_h(i) = data_host.weights[i];
      inv_weights_h(i) = data_host.inv_weights[i];
    }

    auto bil_h = Kokkos::create_mirror_view(boundary_interpolation_left);
    auto bir_h = Kokkos::create_mirror_view(boundary_interpolation_right);
    auto bil_data = detail::GLL::calc_lhat(
        (T)-1.0, std::span<const T, NNodes>{data_host.nodes},
        std::span<const T, NNodes>{data_host.weights});
    auto bir_data = detail::GLL::calc_lhat(
        (T)1.0, std::span<const T, NNodes>{data_host.nodes},
        std::span<const T, NNodes>{data_host.weights});

    for (std::size_t i = 0; i < NNodes; ++i) {
      bil_h(i) = bil_data[i];
      bir_h(i) = bir_data[i];
    }

    auto dm_h = Kokkos::create_mirror_view(derivative_matrix);
    auto ds_h = Kokkos::create_mirror_view(derivative_split);
    auto dst_h = Kokkos::create_mirror_view(derivative_split_transpose);
    auto dd_h = Kokkos::create_mirror_view(derivative_dhat);
    auto ivl_h = Kokkos::create_mirror_view(inverse_vandermonde_legendre);

    auto dm_data = detail::GLL::polynomial_derivative_matrix(
        std::span<const T, NNodes>{data_host.nodes});
    auto ds_data =
        detail::GLL::calc_dsplit(std::span<const T, NNodes>{data_host.nodes},
                                 std::span<const T, NNodes>{data_host.weights});
    auto dd_data =
        detail::GLL::calc_dhat(std::span<const T, NNodes>{data_host.nodes},
                               std::span<const T, NNodes>{data_host.weights});
    auto ivl_data = detail::GLL::inverse_vandermonde_legendre(
        std::span<const T, NNodes>{data_host.nodes});

    for (std::size_t i = 0; i < NNodes; ++i) {
      for (std::size_t j = 0; j < NNodes; ++j) {
        dm_h(i, j) = dm_data(i, j);
        ds_h(i, j) = ds_data(i, j);
        dst_h(i, j) = ds_data(j, i);  // Transpose
        dd_h(i, j) = dd_data(i, j);
        ivl_h(i, j) = ivl_data(i, j);
      }
    }

    Kokkos::deep_copy(nodes, nodes_h);
    Kokkos::deep_copy(weights, weights_h);
    Kokkos::deep_copy(inv_weights, inv_weights_h);
    Kokkos::deep_copy(boundary_interpolation_left, bil_h);
    Kokkos::deep_copy(boundary_interpolation_right, bir_h);
    Kokkos::deep_copy(derivative_matrix, dm_h);
    Kokkos::deep_copy(derivative_split, ds_h);
    Kokkos::deep_copy(derivative_split_transpose, dst_h);
    Kokkos::deep_copy(derivative_dhat, dd_h);
    Kokkos::deep_copy(inverse_vandermonde_legendre, ivl_h);
  }

  static void finalize() {
    nodes = View1D();
    weights = View1D();
    inv_weights = View1D();
    boundary_interpolation_left = View1D();
    boundary_interpolation_right = View1D();
    derivative_matrix = View2D();
    derivative_split = View2D();
    derivative_split_transpose = View2D();
    derivative_dhat = View2D();
    inverse_vandermonde_legendre = View2D();
  }

  static DeviceData device_data() {
    return DeviceData{nodes,
                      weights,
                      inv_weights,
                      boundary_interpolation_left,
                      boundary_interpolation_right,
                      derivative_matrix,
                      derivative_split,
                      derivative_split_transpose,
                      derivative_dhat,
                      inverse_vandermonde_legendre};
  }

  LobattoLegendreBasis() = delete;
};

}  // namespace Basis
}  // namespace DGSEM
