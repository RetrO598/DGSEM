#pragma once

#include "equations/equations_base.hpp"
#include <array>
#include <cstddef>
#include <cstdlib>
namespace DGSEM {
namespace equations {
template <class T>
class InviscidBurgers1D : public Equations1DBase {

public:
  using value_type = T;
  InviscidBurgers1D() {}

  constexpr static std::size_t NDIMS = 1;
  constexpr static std::size_t NVARS = 1;

  T flux(const T &u, std::size_t dim) const { return 0.5 * u * u; }

  std::array<value_type, NVARS> flux(const std::array<value_type, NVARS> &u,
                                     std::size_t idim) const {
    return {0.5 * u[0] * u[0]};
  }

  T get_wave_speed(std::array<value_type, NVARS> u_ll,
                   std::array<value_type, NVARS> u_rr) const {
    value_type ul = u_ll[0];
    value_type ur = u_rr[0];
    return std::max(std::abs(ul), std::abs(ur));
  }
};
} // namespace equations
} // namespace DGSEM