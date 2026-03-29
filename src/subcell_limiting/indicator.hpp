#pragma once

#include <array>
#include <base/base.hpp>
#include <cmath>
#include <cstddef>
#include <equations/equations.hpp>
#include <type_traits>

namespace DGSEM {

template <class T, std::size_t NNodes, std::size_t NDIMS>
struct multiply_scalar_dimensionwise {};

template <class T, std::size_t NNodes>
struct multiply_scalar_dimensionwise<T, NNodes, 1> {
  template <class MatrixType>
  KOKKOS_INLINE_FUNCTION constexpr void static calc(
      std::array<T, NNodes>& data_out, const MatrixType& matrix,
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

template <class T, std::size_t NNodes>
struct multiply_scalar_dimensionwise<T, NNodes, 2> {
  template <class MatrixType>
  KOKKOS_INLINE_FUNCTION constexpr static void calc(
      std::array<std::array<T, NNodes>, NNodes>& data_out,
      const MatrixType& matrix,
      const std::array<std::array<T, NNodes>, NNodes>& data_in) {
    std::array<std::array<T, NNodes>, NNodes> tmp{};

    for (std::size_t j = 0; j < NNodes; ++j) {
      for (std::size_t i = 0; i < NNodes; ++i) {
        T res = 0;
        for (std::size_t ii = 0; ii < NNodes; ++ii) {
          res += matrix(i, ii) * data_in[j][ii];
        }
        tmp[j][i] = res;
      }
    }

    for (std::size_t j = 0; j < NNodes; ++j) {
      for (std::size_t i = 0; i < NNodes; ++i) {
        T res = 0;
        for (std::size_t jj = 0; jj < NNodes; ++jj) {
          res += matrix(j, jj) * tmp[jj][i];
        }
        data_out[j][i] = res;
      }
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

template <class T>
struct IndicatorValueAccessor<equations::CompressibleEuler2D<T>> {
  using traits = equations::EquationTraits<equations::CompressibleEuler2D<T>>;
  using value_type = typename traits::value_type;

  template <class ArrayU>
  KOKKOS_INLINE_FUNCTION static value_type get_indicator_value(
      const ArrayU& u, std::size_t ielem, std::size_t jelem, std::size_t dof) {
    const value_type rho = u(ielem, jelem, dof, 0);
    const value_type rhou = u(ielem, jelem, dof, 1);
    const value_type rhov = u(ielem, jelem, dof, 2);
    const value_type rhoE = u(ielem, jelem, dof, 3);
    const value_type u_vel = rhou / rho;
    const value_type v_vel = rhov / rho;
    const value_type gamma = 1.4;
    const value_type p =
        (gamma - 1.0) * (rhoE - 0.5 * rho * (u_vel * u_vel + v_vel * v_vel));
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
  using BasisData = typename Basis::DeviceData;
  constexpr static std::size_t NDIMS = traits::NDIMS;

  HGIndicator() = delete;

  HGIndicator(value_type alpha_max_, value_type alpha_min_, bool alpha_smooth_,
              BasisData basis_data_)
      : alpha_max(alpha_max_),
        alpha_min(alpha_min_),
        alpha_smooth(alpha_smooth_),
        basis_data(basis_data_) {
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
    auto basis_ = basis_data;
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
              modal, basis_.inverse_vandermonde_legendre, indicator);

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
  BasisData basis_data;

  value_type threshold;
  value_type s;
};

template <class Basis, class Equations>
  requires std::is_base_of<equations::Equations2DBase, Equations>::value
struct HGIndicator<Basis, Equations, 2> {
  using traits = equations::EquationTraits<Equations>;
  using value_type = typename traits::value_type;
  using BasisData = typename Basis::DeviceData;
  constexpr static std::size_t NDIMS = traits::NDIMS;

  HGIndicator() = delete;

  HGIndicator(value_type alpha_max_, value_type alpha_min_, bool alpha_smooth_,
              BasisData basis_data_)
      : alpha_max(alpha_max_),
        alpha_min(alpha_min_),
        alpha_smooth(alpha_smooth_),
        basis_data(basis_data_) {
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
    auto basis_ = basis_data;
    Kokkos::parallel_for(
        "HGIndicator2D",
        Kokkos::MDRangePolicy<Kokkos::Rank<2>>({0, 0},
                                               {n_cells[0], n_cells[1]}),
        KOKKOS_LAMBDA(const int ielem, const int jelem) {
          std::array<std::array<value_type, Basis::NNodes>, Basis::NNodes>
              indicator{};
          std::array<std::array<value_type, Basis::NNodes>, Basis::NNodes>
              modal{};

          auto local_dof = [](std::size_t inode, std::size_t jnode) {
            return jnode * Basis::NNodes + inode;
          };

          for (std::size_t jnode = 0; jnode < Basis::NNodes; ++jnode) {
            for (std::size_t inode = 0; inode < Basis::NNodes; ++inode) {
              const std::size_t dof = local_dof(inode, jnode);
              indicator[jnode][inode] =
                  IndicatorValueAccessor<Equations>::get_indicator_value(
                      u, ielem, jelem, dof);
            }
          }

          multiply_scalar_dimensionwise<value_type, Basis::NNodes, 2>::calc(
              modal, basis_.inverse_vandermonde_legendre, indicator);

          value_type total_energy_clip2 = 0;
          for (std::size_t j = 0; j + 2 < Basis::NNodes; ++j) {
            for (std::size_t i = 0; i + 2 < Basis::NNodes; ++i) {
              total_energy_clip2 += modal[j][i] * modal[j][i];
            }
          }

          value_type total_energy_clip1 = total_energy_clip2;
          for (std::size_t i = 0; i + 1 < Basis::NNodes; ++i) {
            total_energy_clip1 +=
                modal[Basis::NNodes - 2][i] * modal[Basis::NNodes - 2][i];
          }
          for (std::size_t j = 0; j + 2 < Basis::NNodes; ++j) {
            total_energy_clip1 +=
                modal[j][Basis::NNodes - 2] * modal[j][Basis::NNodes - 2];
          }

          value_type total_energy = total_energy_clip1;
          for (std::size_t i = 0; i < Basis::NNodes; ++i) {
            total_energy +=
                modal[Basis::NNodes - 1][i] * modal[Basis::NNodes - 1][i];
          }
          for (std::size_t j = 0; j + 1 < Basis::NNodes; ++j) {
            total_energy +=
                modal[j][Basis::NNodes - 1] * modal[j][Basis::NNodes - 1];
          }

          value_type energy_frac_1 = 0;
          value_type energy_frac_2 = 0;
          if (total_energy != 0) {
            energy_frac_1 = (total_energy - total_energy_clip1) / total_energy;
          }
          if (total_energy_clip1 != 0) {
            energy_frac_2 =
                (total_energy_clip1 - total_energy_clip2) / total_energy_clip1;
          }
          value_type energy =
              energy_frac_1 > energy_frac_2 ? energy_frac_1 : energy_frac_2;

          value_type alpha_e =
              1.0 / (1.0 + exp(-s_ / threshold_ * (energy - threshold_)));

          if (alpha_e < alpha_min_) alpha_e = 0.0;
          if (alpha_e > 1.0 - alpha_min_) alpha_e = 1.0;
          if (alpha_e > alpha_max_) alpha_e = alpha_max_;

          alpha(ielem, jelem) = alpha_e;
        });
  }

 private:
  value_type alpha_max = 0.5;
  value_type alpha_min = 0.001;
  bool alpha_smooth = false;
  BasisData basis_data;

  value_type threshold;
  value_type s;
};
}  // namespace DGSEM
