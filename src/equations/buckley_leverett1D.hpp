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

  KOKKOS_INLINE_FUNCTION
  T flux(const T& u, std::size_t dim) const {
    return static_cast<T>(4.0) * u * u /
           (static_cast<T>(4.0) * u * u +
            (static_cast<T>(1.0) - u) * (static_cast<T>(1.0) - u));
  }

  KOKKOS_INLINE_FUNCTION
  std::array<value_type, NVARS> flux(const std::array<value_type, NVARS>& u,
                                     std::size_t idim) const {
    return {static_cast<value_type>(4.0) * u[0] * u[0] /
            (static_cast<value_type>(4.0) * u[0] * u[0] +
             (static_cast<value_type>(1.0) - u[0]) *
                 (static_cast<value_type>(1.0) - u[0]))};
  }

  KOKKOS_INLINE_FUNCTION
  T get_wave_speed() const { return static_cast<T>(2.35); }
};
} // namespace equations
} // namespace DGSEM
