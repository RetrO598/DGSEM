#pragma once

#include <array>
#include <cstddef>
#include <equations/equations_base.hpp>
namespace DGSEM {
namespace equations {
template <class T>
class BuckleyLeverett1D : public Equations1DBase {

public:
  using value_type = T;
  BuckleyLeverett1D() {}

  constexpr static std::size_t NDIMS = 1;
  constexpr static std::size_t NVARS = 1;

  T flux(const T &u, std::size_t dim) const {
    return 4.0 * u * u / (4.0 * u * u + (1.0 - u) * (1.0 - u));
  }

  std::array<value_type, NVARS> flux(const std::array<value_type, NVARS> &u,
                                     std::size_t idim) const {
    return {4.0 * u[0] * u[0] /
            (4.0 * u[0] * u[0] + (1.0 - u[0]) * (1.0 - u[0]))};
  }

  T get_wave_speed() const { return 2.35; }
};
} // namespace equations
} // namespace DGSEM