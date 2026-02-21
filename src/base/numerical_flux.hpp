#pragma once

#include <array>
#include <cstddef>
#include <cstdlib>
#include <equations/equations.hpp>
namespace DGSEM {
template <class Equations>
struct LaxFriedrichsFlux {};

template <class T>
struct LaxFriedrichsFlux<equations::LinearScalarAdvection1D<T>> {
  using traits =
      equations::EquationTraits<equations::LinearScalarAdvection1D<T>>;
  using value_type = typename traits::value_type;

  constexpr static std::size_t NDIMS = traits::NDIMS;
  constexpr static std::size_t NVARS = traits::NVARS;

  inline constexpr static std::array<value_type, NVARS>
  numerical_flux(const equations::LinearScalarAdvection1D<T> &eq,
                 const std::array<value_type, NVARS> &uL,
                 const std::array<value_type, NVARS> &uR) {
    auto speed = eq.get_wave_speed();
    auto max_speed = std::abs(speed);
    std::array<T, NVARS> flux_L = eq.flux(uL, 0);
    std::array<T, NVARS> flux_R = eq.flux(uR, 0);
    std::array<value_type, NVARS> flux{};
    for (std::size_t i = 0; i < NVARS; ++i) {
      flux[i] =
          0.5 * (flux_L[i] + flux_R[i]) - 0.5 * max_speed * (uR[i] - uL[i]);
    }
    return flux;
  }
};

template <class Equations>
struct CentralFlux {
  using traits = equations::EquationTraits<Equations>;
  using value_type = typename traits::value_type;

  constexpr static std::size_t NDIMS = traits::NDIMS;
  constexpr static std::size_t NVARS = traits::NVARS;

  inline constexpr static std::array<value_type, NVARS>
  numerical_flux(const Equations &eq, const std::array<value_type, NVARS> &uL,
                 const std::array<value_type, NVARS> &uR) {
    auto flux_L = eq.flux(uL, 0);
    auto flux_R = eq.flux(uR, 0);
    std::array<value_type, NVARS> flux{};
    for (std::size_t i = 0; i < NVARS; ++i) {
      flux[i] = 0.5 * (flux_L[i] + flux_R[i]);
    }
    return flux;
  }
};

template <class Equations>
struct flux_godunov {};

template <class T>
struct flux_godunov<equations::LinearScalarAdvection1D<T>> {
  using traits =
      equations::EquationTraits<equations::LinearScalarAdvection1D<T>>;
  constexpr static std::size_t NDIMS = traits::NDIMS;
  constexpr static std::size_t NVARS = traits::NVARS;

  inline constexpr static std::array<T, NVARS>
  numerical_flux(const equations::LinearScalarAdvection1D<T> &eq,
                 const std::array<T, NVARS> &u_ll,
                 const std::array<T, NVARS> &u_rr) {
    auto speed = eq.get_wave_speed();
    if (speed >= 0) {
      return eq.flux(u_ll, 0);
    } else {
      return eq.flux(u_rr, 0);
    }
  }
};
} // namespace DGSEM