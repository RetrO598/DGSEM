#pragma once

#include <array>
#include <cstddef>
namespace DGSEM {
namespace equations {

template <class T>
class LinearScalarAdvection1D {

public:
  using value_type = T;
  LinearScalarAdvection1D(const T &speed_) : speed(speed_) {}

  constexpr static std::size_t NDIMS = 1;
  constexpr static std::size_t NVARS = 1;

  T flux(const T &u, std::size_t dim) const { return speed * u; }

  std::array<value_type, NVARS> flux(const std::array<value_type, NVARS> &u,
                                     std::size_t dim) const {
    return {speed * u[0]};
  }

  T get_wave_speed() const { return speed; }

private:
  T speed{1.0};
};

template <class Equations>
struct EquationTraits {
  using value_type = typename Equations::value_type;
  constexpr static std::size_t NDIMS = Equations::NDIMS;
  constexpr static std::size_t NVARS = Equations::NVARS;
};
} // namespace equations
} // namespace DGSEM
