#pragma once

#include "equations.hpp"
#include <array>
#include <cmath>
#include <cstddef>
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
} // namespace DGSEM
