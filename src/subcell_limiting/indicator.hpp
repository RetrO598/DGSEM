#pragma once

#include "../base/container_fixed.hpp"
#include "equations/compressible_euler1D.hpp"
#include "equations/equations_base.hpp"
#include <array>
#include <cstddef>
#include <equations/equations.hpp>
#include <type_traits>
#include <vector>
#include <xtensor/core/xtensor_forward.hpp>
namespace DGSEM {

template <class T, std::size_t NNodes, std::size_t NDIMS>
struct multiply_scalar_dimensionwise {};

template <class T, std::size_t NNodes>
struct multiply_scalar_dimensionwise<T, NNodes, 1> {
  inline constexpr void static calc(std::array<T, NNodes> &data_out,
                                    const Mat<T, NNodes, NNodes> &matrix,
                                    const std::array<T, NNodes> &data_in) {

    for (size_t i = 0; i < NNodes; ++i) {
      T res = 0; // Initialize the result for data_out[i]
      for (size_t ii = 0; ii < NNodes; ++ii) {
        res += matrix(i, ii) * data_in[ii]; // Matrix multiplication with array
      }
      data_out[i] = res; // Store the result in data_out
    }
  }
};

template <class Equations>
struct IndicatorValueAccessor {
  using traits = equations::EquationTraits<Equations>;
  using value_type = typename traits::value_type;

  inline static value_type get_indicator_value(const xt::xarray<value_type> &u,
                                               std::size_t ielem,
                                               std::size_t inode) {
    return u(ielem, inode, 0);
  }
};

template <class T>
struct IndicatorValueAccessor<equations::CompressibleEuler1D<T>> {
  using traits = equations::EquationTraits<equations::CompressibleEuler1D<T>>;
  using value_type = typename traits::value_type;

  inline static value_type get_indicator_value(const xt::xarray<value_type> &u,
                                               std::size_t ielem,
                                               std::size_t inode) {
    value_type rho = u(ielem, inode, 0);
    value_type mom = u(ielem, inode, 1);
    value_type rhoE = u(ielem, inode, 2);
    value_type u_vel = mom / rho;
    value_type gamma = 1.4;
    value_type p = (gamma - 1.0) * (rhoE - 0.5 * rho * u_vel * u_vel);
    return p * rho;
  }
};

template <class Basis, class Equations>
  requires std::is_base_of<equations::Equations1DBase, Equations>::value
struct HGIndicator {

  using value_type = typename equations::EquationTraits<Equations>::value_type;

  HGIndicator() = delete;

  HGIndicator(value_type alpha_max_, value_type alpha_min_, bool alpha_smooth_)
      : alpha_max(alpha_max_), alpha_min(alpha_min_),
        alpha_smooth(alpha_smooth_) {

    threshold = 0.5 * std::pow(10.0, -1.8 * std::pow(Basis::NNodes, 0.25));
    s = std::log((1.0 - 0.0001) / 0.0001);
  }

  std::vector<value_type> operator()(std::size_t nelem,
                                     const xt::xarray<value_type> &u) {

    std::vector<value_type> alpha(nelem, 0.0);
    for (std::size_t ielem = 0; ielem < nelem; ++ielem) {
      value_type alpha_element = 0.0;
      std::array<value_type, Basis::NNodes> indicator_value{};
      std::array<value_type, Basis::NNodes> modal{};
      for (std::size_t inode = 0; inode < Basis::NNodes; ++inode) {
        value_type u_node =
            IndicatorValueAccessor<Equations>::get_indicator_value(u, ielem,
                                                                   inode);
        indicator_value[inode] = u_node;
      }

      multiply_scalar_dimensionwise<value_type, Basis::NNodes, 1>::calc(
          modal, Basis::inverse_vandermonde_legendre, indicator_value);

      value_type total_energy = 0.0;
      for (std::size_t inode = 0; inode < Basis::NNodes; ++inode) {
        total_energy += modal[inode] * modal[inode];
      }

      value_type total_energy_clip1 = 0.0;
      for (std::size_t inode = 0; inode < Basis::NNodes - 1; ++inode) {
        total_energy_clip1 += modal[inode] * modal[inode];
      }
      value_type total_energy_clip2 = 0.0;
      for (std::size_t inode = 0; inode < Basis::NNodes - 2; ++inode) {
        total_energy_clip2 += modal[inode] * modal[inode];
      }

      value_type energy_frac_1 =
          (total_energy != 0)
              ? (total_energy - total_energy_clip1) / total_energy
              : value_type(0);
      value_type energy_frac_2 =
          (total_energy_clip1 != 0)
              ? (total_energy_clip1 - total_energy_clip2) / total_energy_clip1
              : value_type(0);

      value_type energy = std::max(energy_frac_1, energy_frac_2);

      alpha_element =
          1.0 / (1.0 + std::exp(-s / threshold * (energy - threshold)));

      if (alpha_element < alpha_min) {
        alpha_element = value_type(0.0);
      }

      if (alpha_element > 1.0 - alpha_min) {
        alpha_element = value_type(1.0);
      }

      alpha_element = std::min(alpha_max, alpha_element);
      alpha[ielem] = alpha_element;
    }

    return alpha;
  }

private:
  value_type alpha_max = 0.5;
  value_type alpha_min = 0.001;
  bool alpha_smooth = false;

  value_type threshold;
  value_type s;
};
} // namespace DGSEM