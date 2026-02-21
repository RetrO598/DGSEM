#pragma once

#include "../equations/equations.hpp"
#include <array>
#include <cmath>
#include <cstddef>
#include <cstdlib>
#include <iostream>
#include <numbers>

namespace DGSEM {

template <class Derived, class Equations>
class AbstractInitial {
public:
  using traits = equations::EquationTraits<Equations>;
  using value_type = typename traits::value_type;

  constexpr static std::size_t NDIMS = traits::NDIMS;
  constexpr static std::size_t NVARS = traits::NVARS;

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

  std::array<T, NVARS> operator()(std::array<T, NDIM> coordinate) const {
    std::cout << "doing uniform initialization." << "\n";
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

  std::array<T, NVARS> operator()(std::array<T, NDIMS> coordinate) const {
    T x = coordinate[0];
    T value = 1.0 + 0.5 * std::sin(std::numbers::pi * x);
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
} // namespace DGSEM
