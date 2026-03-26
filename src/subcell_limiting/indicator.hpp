#pragma once

#include <array>
#include <base/base.hpp>
#include <cstddef>
#include <equations/equations.hpp>
#include <type_traits>

#include "equations/equations_base.hpp"

namespace DGSEM {

template <class T, std::size_t NNodes, std::size_t NDIMS>
struct multiply_scalar_dimensionwise {};

template <class T, std::size_t NNodes>
struct multiply_scalar_dimensionwise<T, NNodes, 1> {
  KOKKOS_INLINE_FUNCTION constexpr void static calc(
      std::array<T, NNodes>& data_out, const Mat<T, NNodes, NNodes>& matrix,
      const std::array<T, NNodes>& data_in) {
    for (size_t i = 0; i < NNodes; ++i) {
      T res = 0;  // Initialize the result for data_out[i]
      for (size_t ii = 0; ii < NNodes; ++ii) {
        res += matrix(i, ii) * data_in[ii];  // Matrix multiplication with array
      }
      data_out[i] = res;  // Store the result in data_out
    }
  }
};

template <class Equations>
struct IndicatorValueAccessor {
  using traits = equations::EquationTraits<Equations>;
  using value_type = typename traits::value_type;

  template <class ArrayU>
  KOKKOS_INLINE_FUNCTION static value_type get_indicator_value(
      const ArrayU& u, std::size_t ielem, std::size_t inode) {
    return u(ielem, inode, 0);
  }
};

template <class T>
struct IndicatorValueAccessor<equations::CompressibleEuler1D<T>> {
  using traits = equations::EquationTraits<equations::CompressibleEuler1D<T>>;
  using value_type = typename traits::value_type;

  template <class ArrayU>
  KOKKOS_INLINE_FUNCTION static value_type get_indicator_value(
      const ArrayU& u, std::size_t ielem, std::size_t inode) {
    value_type rho = u(ielem, inode, 0);
    value_type mom = u(ielem, inode, 1);
    value_type rhoE = u(ielem, inode, 2);
    value_type u_vel = mom / rho;
    value_type gamma = 1.4;
    value_type p = (gamma - 1.0) * (rhoE - 0.5 * rho * u_vel * u_vel);
    return p * rho;
  }
};

template <class Basis, class Equations,
          std::size_t NDIMS = equations::EquationTraits<Equations>::NDIMS>
struct HGIndicator;

template <class Basis, class Equations>
  requires std::is_base_of<equations::Equations1DBase, Equations>::value
struct HGIndicator<Basis, Equations, 1> {
  using traits = equations::EquationTraits<Equations>;
  using value_type = typename equations::EquationTraits<Equations>::value_type;
  constexpr static std::size_t NDIMS = traits::NDIMS;

  HGIndicator() = delete;

  HGIndicator(value_type alpha_max_, value_type alpha_min_, bool alpha_smooth_)
      : alpha_max(alpha_max_),
        alpha_min(alpha_min_),
        alpha_smooth(alpha_smooth_) {
    threshold = 0.5 * std::pow(10.0, -1.8 * std::pow(Basis::NNodes, 0.25));
    s = std::log((1.0 - 0.0001) / 0.0001);
  }

  template <class ArrayU, class ElementArray>
  void operator()(const std::array<std::size_t, NDIMS>& n_cells, ArrayU u,
                  ElementArray alpha) {
    auto alpha_max_ = alpha_max;
    auto alpha_min_ = alpha_min;
    auto threshold_ = threshold;
    auto s_ = s;
    Kokkos::parallel_for(
        "HGIndicator", Kokkos::RangePolicy<>(0, n_cells[0]),
        KOKKOS_LAMBDA(const int ielem) {
          std::array<value_type, Basis::NNodes> indicator{};
          std::array<value_type, Basis::NNodes> modal{};

          // 1. gather
          for (int inode = 0; inode < Basis::NNodes; ++inode) {
            indicator[inode] =
                IndicatorValueAccessor<Equations>::get_indicator_value(u, ielem,
                                                                       inode);
          }

          // 2. modal transform
          multiply_scalar_dimensionwise<value_type, Basis::NNodes, 1>::calc(
              modal, Basis::inverse_vandermonde_legendre, indicator);

          // 3. energy
          value_type total = 0;
          value_type clip1 = 0;
          value_type clip2 = 0;

          for (int i = 0; i < Basis::NNodes; ++i) total += modal[i] * modal[i];

          for (int i = 0; i < Basis::NNodes - 1; ++i)
            clip1 += modal[i] * modal[i];

          for (int i = 0; i < Basis::NNodes - 2; ++i)
            clip2 += modal[i] * modal[i];

          value_type e1 = (total != 0) ? (total - clip1) / total : 0;
          value_type e2 = (clip1 != 0) ? (clip1 - clip2) / clip1 : 0;

          value_type energy = e1 > e2 ? e1 : e2;

          value_type alpha_e =
              1.0 / (1.0 + exp(-s_ / threshold_ * (energy - threshold_)));

          if (alpha_e < alpha_min_) alpha_e = 0;

          if (alpha_e > 1.0 - alpha_min_) alpha_e = 1.0;

          if (alpha_e > alpha_max_) alpha_e = alpha_max_;

          alpha(ielem) = alpha_e;
        });
  }

 private:
  value_type alpha_max = 0.5;
  value_type alpha_min = 0.001;
  bool alpha_smooth = false;

  value_type threshold;
  value_type s;
};
}  // namespace DGSEM