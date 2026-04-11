#pragma once

#include <Kokkos_Core.hpp>
#include <cstddef>
namespace DGSEM {
template <class T, std::size_t NVARS>
struct DivergenceChecker {
  bool has_nan;

  KOKKOS_INLINE_FUNCTION
  void init() { has_nan = false; }

  KOKKOS_INLINE_FUNCTION
  void operator()(const Kokkos::Array<T, NVARS>& u) {
    for (std::size_t i = 0; i < NVARS; ++i) {
      if (Kokkos::isnan(u[i])) {
        has_nan = true;
        break;
      }
    }
  }

  KOKKOS_INLINE_FUNCTION
  void join(const DivergenceChecker& other) {
    has_nan = has_nan || other.has_nan;
  }
};
} // namespace DGSEM