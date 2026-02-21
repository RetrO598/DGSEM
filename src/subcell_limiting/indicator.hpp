#pragma once

#include "../base/container_fixed.hpp"
#include <array>
#include <cstddef>
#include <equations/equations.hpp>
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

template <class Basis, class Equations>
struct HGIndicator {};

template <class Basis, class T>
struct HGIndicator<Basis, equations::LinearScalarAdvection1D<T>> {

  HGIndicator() = delete;

  HGIndicator(T alpha_max_, T alpha_min_, bool alpha_smooth_)
      : alpha_max(alpha_max_), alpha_min(alpha_min_),
        alpha_smooth(alpha_smooth_) {

    threshold = 0.5 * std::pow(10.0, -1.8 * std::pow(Basis::NNodes, 0.25));
    s = std::log((1.0 - 0.0001) / 0.0001);
  }

  std::vector<T> operator()(std::size_t nelem, const xt::xarray<T> &u) {

    std::vector<T> alpha(nelem, 0.0);
    for (std::size_t ielem = 0; ielem < nelem; ++ielem) {
      T alpha_element = 0.0;
      std::array<T, Basis::NNodes> indicator_value{};
      std::array<T, Basis::NNodes> modal{};
      for (std::size_t inode = 0; inode < Basis::NNodes; ++inode) {
        T u_node = u(ielem, inode, 0);
        indicator_value[inode] = u_node;
      }

      multiply_scalar_dimensionwise<T, Basis::NNodes, 1>::calc(
          modal, Basis::inverse_vandermonde_legendre, indicator_value);

      T total_energy = 0.0;
      for (std::size_t inode = 0; inode < Basis::NNodes; ++inode) {
        total_energy += modal[inode] * modal[inode];
      }

      T total_energy_clip1 = 0.0;
      for (std::size_t inode = 0; inode < Basis::NNodes - 1; ++inode) {
        total_energy_clip1 += modal[inode] * modal[inode];
      }
      T total_energy_clip2 = 0.0;
      for (std::size_t inode = 0; inode < Basis::NNodes - 2; ++inode) {
        total_energy_clip2 += modal[inode] * modal[inode];
      }

      T energy_frac_1 = (total_energy != 0)
                            ? (total_energy - total_energy_clip1) / total_energy
                            : T(0);
      T energy_frac_2 =
          (total_energy_clip1 != 0)
              ? (total_energy_clip1 - total_energy_clip2) / total_energy_clip1
              : T(0);

      // Calculate energy as the maximum of energy_frac_1 and energy_frac_2
      T energy = std::max(energy_frac_1, energy_frac_2);

      // Calculate alpha_element based on energy and threshold using a sigmoid
      // function
      alpha_element =
          1.0 / (1.0 + std::exp(-s / threshold * (energy - threshold)));

      // Ensure alpha_element is within the bounds for DG and FV
      if (alpha_element < alpha_min) {
        alpha_element = T(0.0);
      }

      if (alpha_element > 1.0 - alpha_min) {
        alpha_element = T(1.0);
      }

      // Clip the maximum allowed FV value for alpha_element
      alpha_element = std::min(alpha_max, alpha_element);
      alpha[ielem] = alpha_element;
    }

    return alpha;
  }

private:
  T alpha_max = 0.5;
  T alpha_min = 0.001;
  bool alpha_smooth = false;

  T threshold;
  T s;
};
} // namespace DGSEM