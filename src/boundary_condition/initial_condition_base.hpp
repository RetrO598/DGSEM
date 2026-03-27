#pragma once

#include <Kokkos_Core.hpp>
#include <Kokkos_Macros.hpp>
#include <array>
#include <cmath>
#include <cstddef>
#include <cstdlib>
#include <equations/equations.hpp>
#include <numbers>

namespace DGSEM {

template <class Derived, class Equations>
class AbstractInitial {
public:
  using traits = equations::EquationTraits<Equations>;
  using value_type = typename traits::value_type;

  constexpr static std::size_t NDIMS = traits::NDIMS;
  constexpr static std::size_t NVARS = traits::NVARS;

  KOKKOS_INLINE_FUNCTION
  std::array<value_type, NVARS>
  get_initial(std::array<value_type, NDIMS> coord) const {
    return static_cast<const Derived *>(this)->operator()(coord);
  }
};

template <class Equations>
class UniformInitial;

template <class T>
class UniformInitial<equations::LinearScalarAdvection1D<T>>
    : public AbstractInitial<
          UniformInitial<equations::LinearScalarAdvection1D<T>>,
          equations::LinearScalarAdvection1D<T>> {
public:
  using Eq = equations::LinearScalarAdvection1D<T>;
  using Base = AbstractInitial<UniformInitial<Eq>, Eq>;
  using value_type = T;
  inline constexpr static std::size_t NDIM = Eq::NDIMS;
  inline constexpr static std::size_t NVARS = Eq::NVARS;

  UniformInitial(std::array<value_type, NVARS> vars) : initial_vars(vars) {}

  KOKKOS_INLINE_FUNCTION
  std::array<T, NVARS> operator()(std::array<T, NDIM> coordinate) const {
    return initial_vars;
  }

private:
  std::array<T, NVARS> initial_vars;
};

template <class T>
class SinwaveInitial
    : public AbstractInitial<SinwaveInitial<T>,
                             equations::LinearScalarAdvection1D<T>> {
public:
  using Eq = equations::LinearScalarAdvection1D<T>;
  using Base =
      AbstractInitial<SinwaveInitial<T>, equations::LinearScalarAdvection1D<T>>;
  using value_type = T;
  inline constexpr static std::size_t NDIMS = Eq::NDIMS;
  inline constexpr static std::size_t NVARS = Eq::NVARS;

  SinwaveInitial() = default;

  KOKKOS_INLINE_FUNCTION
  std::array<T, NVARS> operator()(std::array<T, NDIMS> coordinate) const {
    T x = coordinate[0];
    T value = 1.0 + 0.5 * std::sin(std::numbers::pi * x);
    return {value};
  }
};

template <class T>
class BurgersSinwaveInitial
    : public AbstractInitial<BurgersSinwaveInitial<T>,
                             equations::InviscidBurgers1D<T>> {
public:
  using Eq = equations::InviscidBurgers1D<T>;
  using Base = AbstractInitial<BurgersSinwaveInitial<T>,
                               equations::InviscidBurgers1D<T>>;
  using value_type = T;
  inline constexpr static std::size_t NDIMS = Eq::NDIMS;
  inline constexpr static std::size_t NVARS = Eq::NVARS;

  BurgersSinwaveInitial() = default;

  KOKKOS_INLINE_FUNCTION
  std::array<T, NVARS> operator()(std::array<T, NDIMS> coordinate) const {
    T x = coordinate[0];
    T value = std::sin(2.0 * std::numbers::pi * x);
    return {value};
  }
};

template <class T>
class GaussianInitial
    : public AbstractInitial<GaussianInitial<T>,
                             equations::LinearScalarAdvection1D<T>> {
public:
  using Eq = equations::LinearScalarAdvection1D<T>;
  using Base = AbstractInitial<GaussianInitial<T>,
                               equations::LinearScalarAdvection1D<T>>;
  using value_type = T;
  inline constexpr static std::size_t NDIMS = Eq::NDIMS;
  inline constexpr static std::size_t NVARS = Eq::NVARS;

  GaussianInitial() = default;

  KOKKOS_INLINE_FUNCTION
  std::array<T, NVARS> operator()(std::array<T, NDIMS> coordinate) const {
    T x = coordinate[0];
    T value = std::exp(-50.0 * x * x);
    return {value};
  }
};

template <class T>
class CompositeWaveInitial
    : public AbstractInitial<CompositeWaveInitial<T>,
                             equations::LinearScalarAdvection1D<T>> {
public:
  using Eq = equations::LinearScalarAdvection1D<T>;
  using Base = AbstractInitial<CompositeWaveInitial<T>,
                               equations::LinearScalarAdvection1D<T>>;
  using value_type = T;
  inline constexpr static std::size_t NDIMS = Eq::NDIMS;
  inline constexpr static std::size_t NVARS = Eq::NVARS;

  CompositeWaveInitial() = default;

  KOKKOS_INLINE_FUNCTION
  std::array<T, NVARS> operator()(std::array<T, NDIMS> coordinate) const {
    T xmin = -1.0;
    T xmax = 1.0;
    T x = coordinate[0];

    T value = 0.0;
    if (x > -0.8 && x < -0.6) {
      value = std::exp(-std::log(2.0) * (x + 0.7) * (x + 0.7) / 0.0009);
    } else if (x > -0.4 && x < -0.2) {
      value = 1.0;
    } else if (x > 0.0 && x < 0.2) {
      value = 1.0 - std::abs(10.0 * (x - 0.1));
    } else if (x > 0.4 && x < 0.6) {
      value = std::sqrt(1.0 - 100.0 * (x - 0.5) * (x - 0.5));
    } else {
      value = 0.0;
    }
    return {value};
  }
};

template <class T>
class BuckleyLeverettInitial
    : public AbstractInitial<BuckleyLeverettInitial<T>,
                             equations::BuckleyLeverett1D<T>> {
public:
  using Eq = equations::BuckleyLeverett1D<T>;
  using Base = AbstractInitial<BuckleyLeverettInitial<T>,
                               equations::BuckleyLeverett1D<T>>;
  using value_type = T;
  inline constexpr static std::size_t NDIMS = Eq::NDIMS;
  inline constexpr static std::size_t NVARS = Eq::NVARS;

  BuckleyLeverettInitial() = default;

  KOKKOS_INLINE_FUNCTION
  std::array<T, NVARS> operator()(std::array<T, NDIMS> coordinate) const {
    T x = coordinate[0];
    T value = 0.0;
    if (x > -0.5 && x < 0.0) {
      value = 1.0;
    } else {
      value = 0.0;
    }
    return {value};
  }
};

template <class T>
class ShuOsherInitial
    : public AbstractInitial<ShuOsherInitial<T>,
                             equations::CompressibleEuler1D<T>> {
public:
  using Eq = equations::CompressibleEuler1D<T>;
  using Base =
      AbstractInitial<ShuOsherInitial<T>, equations::CompressibleEuler1D<T>>;
  using value_type = T;
  inline constexpr static std::size_t NDIMS = Eq::NDIMS;
  inline constexpr static std::size_t NVARS = Eq::NVARS;

  ShuOsherInitial() = default;

  KOKKOS_INLINE_FUNCTION
  std::array<T, NVARS> operator()(std::array<T, NDIMS> coordinate) const {
    T x = coordinate[0];
    T rho = 0.0;
    T v = 0.0;
    T p = 0.0;
    if (x < -4.0) {
      rho = 3.857;
      v = 2.629;
      p = 10.333;
    } else if (x >= -4.0) {
      rho = 1.0 + 0.2 * std::sin(5.0 * x);
      v = 0.0;
      p = 1.0;
    }
    return {rho, rho * v, rho * 0.5 * v * v + p / (1.4 - 1.0)};
  }
};
// Sod Shock Tube Initial Condition for 1D Euler
template <class T>
class SodShockTubeInitial
    : public AbstractInitial<SodShockTubeInitial<T>,
                             equations::CompressibleEuler1D<T>> {
public:
  using Eq = equations::CompressibleEuler1D<T>;
  using Base = AbstractInitial<SodShockTubeInitial<T>,
                               equations::CompressibleEuler1D<T>>;
  using value_type = T;
  inline constexpr static std::size_t NDIMS = Eq::NDIMS;
  inline constexpr static std::size_t NVARS = Eq::NVARS;

  SodShockTubeInitial() = default;

  KOKKOS_INLINE_FUNCTION
  std::array<T, NVARS> operator()(std::array<T, NDIMS> coordinate) const {
    T x = coordinate[0];
    T rho, u, p;
    if (x < 0.5) {
      rho = 1.0;
      u = 0.0;
      p = 1.0;
    } else {
      rho = 0.125;
      u = 0.0;
      p = 0.1;
    }
    T mom = rho * u;
    T gamma = 1.4;
    T rhoE = p / (gamma - 1.0) + 0.5 * rho * u * u;
    return {rho, mom, rhoE};
  }
};
} // namespace DGSEM
