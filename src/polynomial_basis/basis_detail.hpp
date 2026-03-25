#pragma once
#include <array>
#include <base/base.hpp>
#include <cmath>
#include <cstddef>
#include <cstdlib>
#include <limits>
#include <numbers>
#include <numeric>
#include <span>
#include <tuple>
#include <utility>


namespace DGSEM {
namespace Basis {

namespace detail {
namespace GLL {
template <class T, std::size_t N>
struct Data {
  std::array<T, N> nodes{};
  std::array<T, N> weights{};
  std::array<T, N> inv_weights{};
};

template <class T>
static constexpr std::tuple<T, T, T> evaluate_q_and_l(std::size_t polydeg_,
                                                      T x) {
  T L_Nm2 = (T)1.0;
  T L_Nm1 = x;

  T Lder_Nm2 = (T)0.0;
  T Lder_Nm1 = (T)1.0;

  T L_N = 0.0;
  T Lder_N = 0.0;
  for (std::size_t i = 2; i <= polydeg_; ++i) {
    L_N = ((T)(2.0 * i - 1.0) * x * L_Nm1 - (T)(i - 1.0) * L_Nm2) / (T)i;
    Lder_N = Lder_Nm2 + (T)(2.0 * i - 1.0) * L_Nm1;
    L_Nm2 = L_Nm1;
    L_Nm1 = L_N;
    Lder_Nm2 = Lder_Nm1;
    Lder_Nm1 = Lder_N;
  }

  T q = (T)(2.0 * polydeg_ + 1.0) / (polydeg_ + 1.0) * (x * L_N - L_Nm2);
  T qder = (2.0 * polydeg_ + 1.0) * L_N;

  return {q, qder, L_N};
}

template <class T, std::size_t NNodes>
static Data<T, NNodes> gauss_lobatto_nodes_weights() {
  std::size_t n_iterations = 20;
  std::size_t Polydeg = NNodes - 1;
  T tolerance = 2.0 * std::numeric_limits<T>::epsilon();

  Data<T, NNodes> data{};

  if (NNodes == 1) {
    data.nodes[0] = 0;
    data.weights[0] = 2;
    data.inv_weights[0] = 1 / data.weights[0];
    return data;
  }

  data.nodes[0] = -1.0;
  data.nodes[NNodes - 1] = 1.0;
  data.weights[0] = (T)2.0 / (Polydeg * (Polydeg + 1.0));
  data.weights[NNodes - 1] = data.weights[0];

  if (NNodes > 1) {
    auto cont1 = (T)std::numbers::pi / Polydeg;
    auto cont2 = (T)3.0 / ((T)8.0 * Polydeg * (T)std::numbers::pi);

    for (std::size_t i = 0; i < (std::size_t)(NNodes / 2 - 1); ++i) {
      data.nodes[i + 1] =
          -std::cos(cont1 * ((i + 1) + 0.25) - cont2 / ((i + 1) + 0.25));

      for (std::size_t n = 0; n < n_iterations; ++n) {
        auto [q, qder, L_N] = evaluate_q_and_l(Polydeg, data.nodes[i + 1]);
        T dx = -q / qder;
        data.nodes[i + 1] += dx;

        if (std::abs(dx) < tolerance * std::abs(data.nodes[i + 1])) {
          break;
        }
      }

      auto [q, qder, L_N] = evaluate_q_and_l(Polydeg, data.nodes[i + 1]);
      data.weights[i + 1] = data.weights[0] / (L_N * L_N);

      data.nodes[NNodes - 2 - i] = -data.nodes[i + 1];
      data.weights[NNodes - 2 - i] = data.weights[i + 1];
    }
  }

  if (NNodes % 2 == 1) {
    data.nodes[NNodes / 2] = 0.0;
    auto [q, qder, L_N] = evaluate_q_and_l(Polydeg, 0.0);
    data.weights[NNodes / 2] = data.weights[0] / (L_N * L_N);
  }

  for (std::size_t i = 0; i < NNodes; ++i) {
    data.inv_weights[i] = (T)1.0 / data.weights[i];
  }

  return data;
}

template <class T, std::size_t NNodes>
std::array<T, NNodes> barycenteric_weights(std::span<const T, NNodes> nodes) {
  std::array<T, NNodes> weights{};
  weights.fill(1.0);

  for (std::size_t j = 1; j < NNodes; ++j) {
    for (std::size_t k = 0; k < j; ++k) {
      weights[k] *= (nodes[k] - nodes[j]);
      weights[j] *= (nodes[j] - nodes[k]);
    }
  }

  for (auto &weight : weights) {
    weight = 1.0 / weight;
  }

  return weights;
}

template <class T, std::size_t NNodes>
std::array<T, NNodes>
lagrange_interpolating_polynomials(T x, std::span<const T, NNodes> nodes,
                                   std::span<const T, NNodes> wbary) {
  std::array<T, NNodes> polynomials{};

  for (std::size_t i = 0; i < NNodes; ++i) {
    if (std::abs(x - nodes[i]) < std::numeric_limits<T>::epsilon()) {
      polynomials[i] = 1.0;
      return polynomials;
    }
  }

  for (std::size_t i = 0; i < NNodes; ++i) {
    polynomials[i] = wbary[i] / (x - nodes[i]);
  }

  T total = std::accumulate(polynomials.begin(), polynomials.end(), 0.0);

  for (auto &poly : polynomials) {
    poly /= total;
  }

  return polynomials;
}

template <class T, std::size_t NNodes>
std::array<T, NNodes> calc_lhat(T x, std::span<const T, NNodes> nodes,
                                std::span<const T, NNodes> weights) {
  auto wbary = barycenteric_weights(nodes);
  auto lhat = lagrange_interpolating_polynomials(
      x, nodes, std::span<const T, NNodes>{wbary});

  for (std::size_t i = 0; i < NNodes; ++i) {
    lhat[i] /= weights[i];
  }

  return lhat;
}

template <class T, std::size_t NNodes>
Mat<T, NNodes, NNodes>
polynomial_derivative_matrix(std::span<const T, NNodes> nodes) {
  Mat<T, NNodes, NNodes> derivative_matrix{};

  auto wbary = barycenteric_weights(nodes);

  for (std::size_t i = 0; i < NNodes; ++i) {
    for (std::size_t j = 0; j < NNodes; ++j) {
      if (i != j) {
        derivative_matrix(i, j) =
            wbary[j] / wbary[i] * 1.0 / (nodes[i] - nodes[j]);
        derivative_matrix(i, i) -= derivative_matrix(i, j);
      }
    }
  }

  return derivative_matrix;
}

template <class T, std::size_t NNodes>
Mat<T, NNodes, NNodes> calc_dsplit(std::span<const T, NNodes> nodes,
                                   std::span<const T, NNodes> weights) {
  Mat<T, NNodes, NNodes> dsplit{};
  dsplit = (T)2.0 * polynomial_derivative_matrix(nodes);

  dsplit(0, 0) += 1.0 / weights[0];
  dsplit(NNodes - 1, NNodes - 1) -= 1.0 / weights[NNodes - 1];

  return dsplit;
}

template <class T, std::size_t NNodes>
Mat<T, NNodes, NNodes> calc_dhat(std::span<const T, NNodes> nodes,
                                 std::span<const T, NNodes> weights) {
  Mat<T, NNodes, NNodes> dhat{};
  dhat = polynomial_derivative_matrix(nodes).transpose();

  for (std::size_t i = 0; i < NNodes; ++i) {
    for (std::size_t j = 0; j < NNodes; ++j) {
      dhat(j, i) *= -weights[i] / weights[j];
    }
  }

  return dhat;
}

template <class T>
std::pair<T, T> legendre_polynomial_and_derivative(std::size_t n, T x) {
  T poly = 0.0;
  T deriv = 0.0;
  if (n == 0) {
    poly = 1.0;
    deriv = 0.0;
  } else if (n == 1) {
    poly = x;
    deriv = 1.0;
  } else {
    T Poly_Nm2 = 1.0;
    T Poly_Nm1 = x;
    T Deriv_Nm2 = 0.0;
    T Deriv_Nm1 = 1.0;

    for (std::size_t i = 2; i <= n; ++i) {
      poly =
          ((T)(2.0 * i - 1.0) * x * Poly_Nm1 - (T)(i - 1.0) * Poly_Nm2) / (T)i;
      deriv = Deriv_Nm2 + (T)(2.0 * i - 1.0) * Poly_Nm1;
      Poly_Nm2 = Poly_Nm1;
      Poly_Nm1 = poly;
      Deriv_Nm2 = Deriv_Nm1;
      Deriv_Nm1 = deriv;
    }
  }

  poly = poly * std::sqrt(n + 0.5);
  deriv = deriv * std::sqrt(n + 0.5);

  return {poly, deriv};
}

template <class T, std::size_t NNodes>
Mat<T, NNodes, NNodes>
inverse_vandermonde_legendre(std::span<const T, NNodes> nodes) {
  Mat<T, NNodes, NNodes> inverse_vandermonde{};
  Mat<T, NNodes, NNodes> vandermonde{};
  for (std::size_t i = 0; i < NNodes; ++i) {
    for (std::size_t j = 0; j < NNodes; ++j) {
      vandermonde(i, j) = legendre_polynomial_and_derivative(j, nodes[i]).first;
    }
  }
  inverse_vandermonde = vandermonde.inverse();
  return inverse_vandermonde;
}
} // namespace GLL
} // namespace detail
} // namespace Basis
} // namespace DGSEM
