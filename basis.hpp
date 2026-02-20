#pragma once
#include <array>
#include <cmath>
#include <cstddef>
#include <span>

#include "basis_detail.hpp"
#include "container_fixed.hpp"
namespace DGSEM {
namespace Basis {

template <class T, std::size_t Polydeg>
class LobattoLegendreBasis {
public:
  static constexpr std::size_t NNodes = Polydeg + 1;
  inline static detail::GLL::Data<T, NNodes> data =
      detail::GLL::gauss_lobatto_nodes_weights<T, NNodes>();

  inline static std::array<T, NNodes> nodes = data.nodes;
  inline static std::array<T, NNodes> weights = data.weights;
  inline static std::array<T, NNodes> inv_weights = data.inv_weights;

  inline static std::array<T, NNodes> boundary_interpolation_left =
      detail::GLL::calc_lhat((T)-1.0, std::span<const T, NNodes>{nodes},
                             std::span<const T, NNodes>{weights});

  inline static std::array<T, NNodes> boundary_interpolation_right =
      detail::GLL::calc_lhat((T)1.0, std::span<const T, NNodes>{nodes},
                             std::span<const T, NNodes>{weights});

  inline static Mat<T, NNodes, NNodes> derivative_matrix =
      detail::GLL::polynomial_derivative_matrix(
          std::span<const T, NNodes>{nodes});

  inline static Mat<T, NNodes, NNodes> derivative_split =
      detail::GLL::calc_dsplit(std::span<const T, NNodes>{nodes},
                               std::span<const T, NNodes>{weights});

  inline static Mat<T, NNodes, NNodes> derivative_split_transpose =
      derivative_split.transpose();

  inline static Mat<T, NNodes, NNodes> derivative_dhat = detail::GLL::calc_dhat(
      std::span<const T, NNodes>{nodes}, std::span<const T, NNodes>{weights});

  inline static Mat<T, NNodes, NNodes> inverse_vandermonde_legendre =
      detail::GLL::inverse_vandermonde_legendre(
          std::span<const T, NNodes>{nodes});

  LobattoLegendreBasis() = delete;
};

} // namespace Basis
} // namespace DGSEM